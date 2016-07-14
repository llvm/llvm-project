//===------ PPCGCodeGeneration.cpp - Polly Accelerator Code Generation. ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Take a scop created by ScopInfo and map it to GPU code using the ppcg
// GPU mapping strategy.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/IslNodeBuilder.h"
#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"

#include "isl/union_map.h"

extern "C" {
#include "gpu.h"
#include "ppcg.h"
}

#include "llvm/Support/Debug.h"

using namespace polly;
using namespace llvm;

#define DEBUG_TYPE "polly-codegen-ppcg"

static cl::opt<bool> DumpSchedule("polly-acc-dump-schedule",
                                  cl::desc("Dump the computed GPU Schedule"),
                                  cl::Hidden, cl::init(false), cl::ZeroOrMore,
                                  cl::cat(PollyCategory));

namespace {
class PPCGCodeGeneration : public ScopPass {
public:
  static char ID;

  /// The scop that is currently processed.
  Scop *S;

  PPCGCodeGeneration() : ScopPass(ID) {}

  /// Construct compilation options for PPCG.
  ///
  /// @returns The compilation options.
  ppcg_options *createPPCGOptions() {
    auto DebugOptions =
        (ppcg_debug_options *)malloc(sizeof(ppcg_debug_options));
    auto Options = (ppcg_options *)malloc(sizeof(ppcg_options));

    DebugOptions->dump_schedule_constraints = false;
    DebugOptions->dump_schedule = false;
    DebugOptions->dump_final_schedule = false;
    DebugOptions->dump_sizes = false;

    Options->debug = DebugOptions;

    Options->reschedule = true;
    Options->scale_tile_loops = false;
    Options->wrap = false;

    Options->non_negative_parameters = false;
    Options->ctx = nullptr;
    Options->sizes = nullptr;

    Options->tile_size = 32;

    Options->use_private_memory = false;
    Options->use_shared_memory = false;
    Options->max_shared_memory = 0;

    Options->target = PPCG_TARGET_CUDA;
    Options->openmp = false;
    Options->linearize_device_arrays = true;
    Options->live_range_reordering = false;

    Options->opencl_compiler_options = nullptr;
    Options->opencl_use_gpu = false;
    Options->opencl_n_include_file = 0;
    Options->opencl_include_files = nullptr;
    Options->opencl_print_kernel_types = false;
    Options->opencl_embed_kernel_code = false;

    Options->save_schedule_file = nullptr;
    Options->load_schedule_file = nullptr;

    return Options;
  }

  /// Get a tagged access relation containing all accesses of type @p AccessTy.
  ///
  /// Instead of a normal access of the form:
  ///
  ///   Stmt[i,j,k] -> Array[f_0(i,j,k), f_1(i,j,k)]
  ///
  /// a tagged access has the form
  ///
  ///   [Stmt[i,j,k] -> id[]] -> Array[f_0(i,j,k), f_1(i,j,k)]
  ///
  /// where 'id' is an additional space that references the memory access that
  /// triggered the access.
  ///
  /// @param AccessTy The type of the memory accesses to collect.
  ///
  /// @return The relation describing all tagged memory accesses.
  isl_union_map *getTaggedAccesses(enum MemoryAccess::AccessType AccessTy) {
    isl_union_map *Accesses = isl_union_map_empty(S->getParamSpace());

    for (auto &Stmt : *S)
      for (auto &Acc : Stmt)
        if (Acc->getType() == AccessTy) {
          isl_map *Relation = Acc->getAccessRelation();
          Relation = isl_map_intersect_domain(Relation, Stmt.getDomain());

          isl_space *Space = isl_map_get_space(Relation);
          Space = isl_space_range(Space);
          Space = isl_space_from_range(Space);
          isl_map *Universe = isl_map_universe(Space);
          Relation = isl_map_domain_product(Relation, Universe);
          Accesses = isl_union_map_add_map(Accesses, Relation);
        }

    return Accesses;
  }

  /// Get the set of all read accesses, tagged with the access id.
  ///
  /// @see getTaggedAccesses
  isl_union_map *getTaggedReads() {
    return getTaggedAccesses(MemoryAccess::READ);
  }

  /// Get the set of all may (and must) accesses, tagged with the access id.
  ///
  /// @see getTaggedAccesses
  isl_union_map *getTaggedMayWrites() {
    return isl_union_map_union(getTaggedAccesses(MemoryAccess::MAY_WRITE),
                               getTaggedAccesses(MemoryAccess::MUST_WRITE));
  }

  /// Get the set of all must accesses, tagged with the access id.
  ///
  /// @see getTaggedAccesses
  isl_union_map *getTaggedMustWrites() {
    return getTaggedAccesses(MemoryAccess::MUST_WRITE);
  }

  /// Collect parameter and array names as isl_ids.
  ///
  /// To reason about the different parameters and arrays used, ppcg requires
  /// a list of all isl_ids in use. As PPCG traditionally performs
  /// source-to-source compilation each of these isl_ids is mapped to the
  /// expression that represents it. As we do not have a corresponding
  /// expression in Polly, we just map each id to a 'zero' expression to match
  /// the data format that ppcg expects.
  ///
  /// @returns Retun a map from collected ids to 'zero' ast expressions.
  __isl_give isl_id_to_ast_expr *getNames() {
    auto *Names = isl_id_to_ast_expr_alloc(
        S->getIslCtx(),
        S->getNumParams() + std::distance(S->array_begin(), S->array_end()));
    auto *Zero = isl_ast_expr_from_val(isl_val_zero(S->getIslCtx()));
    auto *Space = S->getParamSpace();

    for (int I = 0, E = S->getNumParams(); I < E; ++I) {
      isl_id *Id = isl_space_get_dim_id(Space, isl_dim_param, I);
      Names = isl_id_to_ast_expr_set(Names, Id, isl_ast_expr_copy(Zero));
    }

    for (auto &Array : S->arrays()) {
      auto Id = Array.second->getBasePtrId();
      Names = isl_id_to_ast_expr_set(Names, Id, isl_ast_expr_copy(Zero));
    }

    isl_space_free(Space);
    isl_ast_expr_free(Zero);

    return Names;
  }

  /// Create a new PPCG scop from the current scop.
  ///
  /// The PPCG scop is initialized with data from the current polly::Scop. From
  /// this initial data, the data-dependences in the PPCG scop are initialized.
  /// We do not use Polly's dependence analysis for now, to ensure we match
  /// the PPCG default behaviour more closely.
  ///
  /// @returns A new ppcg scop.
  ppcg_scop *createPPCGScop() {
    auto PPCGScop = (ppcg_scop *)malloc(sizeof(ppcg_scop));

    PPCGScop->options = createPPCGOptions();

    PPCGScop->start = 0;
    PPCGScop->end = 0;

    PPCGScop->context = S->getContext();
    PPCGScop->domain = S->getDomains();
    PPCGScop->call = nullptr;
    PPCGScop->tagged_reads = getTaggedReads();
    PPCGScop->reads = S->getReads();
    PPCGScop->live_in = nullptr;
    PPCGScop->tagged_may_writes = getTaggedMayWrites();
    PPCGScop->may_writes = S->getWrites();
    PPCGScop->tagged_must_writes = getTaggedMustWrites();
    PPCGScop->must_writes = S->getMustWrites();
    PPCGScop->live_out = nullptr;
    PPCGScop->tagged_must_kills = isl_union_map_empty(S->getParamSpace());
    PPCGScop->tagger = nullptr;

    PPCGScop->independence = nullptr;
    PPCGScop->dep_flow = nullptr;
    PPCGScop->tagged_dep_flow = nullptr;
    PPCGScop->dep_false = nullptr;
    PPCGScop->dep_forced = nullptr;
    PPCGScop->dep_order = nullptr;
    PPCGScop->tagged_dep_order = nullptr;

    PPCGScop->schedule = S->getScheduleTree();
    PPCGScop->names = getNames();

    PPCGScop->pet = nullptr;

    compute_tagger(PPCGScop);
    compute_dependences(PPCGScop);

    return PPCGScop;
  }

  /// Create a default-initialized PPCG GPU program.
  ///
  /// @returns A new gpu grogram description.
  gpu_prog *createPPCGProg(ppcg_scop *PPCGScop) {

    if (!PPCGScop)
      return nullptr;

    auto PPCGProg = isl_calloc_type(S->getIslCtx(), struct gpu_prog);

    PPCGProg->ctx = S->getIslCtx();
    PPCGProg->scop = PPCGScop;
    PPCGProg->context = isl_set_copy(PPCGScop->context);
    PPCGProg->read = nullptr;
    PPCGProg->may_write = nullptr;
    PPCGProg->must_write = nullptr;
    PPCGProg->tagged_must_kill = nullptr;
    PPCGProg->may_persist = nullptr;
    PPCGProg->to_outer = nullptr;
    PPCGProg->to_inner = nullptr;
    PPCGProg->any_to_outer = nullptr;
    PPCGProg->array_order = nullptr;
    PPCGProg->n_stmts = 0;
    PPCGProg->stmts = nullptr;
    PPCGProg->n_array = 0;
    PPCGProg->array = nullptr;

    return PPCGProg;
  }

  // Generate a GPU program using PPCG.
  //
  // GPU mapping consists of multiple steps:
  //
  //  1) Compute new schedule for the program.
  //  2) Map schedule to GPU (TODO)
  //  3) Generate code for new schedule (TODO)
  //
  // We do not use here the Polly ScheduleOptimizer, as the schedule optimizer
  // is mostly CPU specific. Instead, we use PPCG's GPU code generation
  // strategy directly from this pass.
  gpu_gen *generateGPU(ppcg_scop *PPCGScop, gpu_prog *PPCGProg) {

    auto PPCGGen = isl_calloc_type(S->getIslCtx(), struct gpu_gen);

    PPCGGen->ctx = S->getIslCtx();
    PPCGGen->options = PPCGScop->options;
    PPCGGen->print = nullptr;
    PPCGGen->print_user = nullptr;
    PPCGGen->prog = PPCGProg;
    PPCGGen->tree = nullptr;
    PPCGGen->types.n = 0;
    PPCGGen->types.name = nullptr;
    PPCGGen->sizes = nullptr;
    PPCGGen->used_sizes = nullptr;
    PPCGGen->kernel_id = 0;

    // Set scheduling strategy to same strategy PPCG is using.
    isl_options_set_schedule_outer_coincidence(PPCGGen->ctx, true);
    isl_options_set_schedule_maximize_band_depth(PPCGGen->ctx, true);

    isl_schedule *Schedule = get_schedule(PPCGGen);

    int has_permutable = has_any_permutable_node(Schedule);

    if (!has_permutable || has_permutable < 0)
      Schedule = isl_schedule_free(Schedule);
    else
      Schedule = map_to_device(PPCGGen, Schedule);

    if (DumpSchedule) {
      isl_printer *P = isl_printer_to_str(S->getIslCtx());
      P = isl_printer_set_yaml_style(P, ISL_YAML_STYLE_BLOCK);
      P = isl_printer_print_str(P, "Schedule\n");
      P = isl_printer_print_str(P, "========\n");
      if (Schedule)
        P = isl_printer_print_schedule(P, Schedule);
      else
        P = isl_printer_print_str(P, "No schedule found\n");

      printf("%s\n", isl_printer_get_str(P));
      isl_printer_free(P);
    }

    isl_schedule_free(Schedule);

    return PPCGGen;
  }

  /// Free gpu_gen structure.
  ///
  /// @param PPCGGen The ppcg_gen object to free.
  void freePPCGGen(gpu_gen *PPCGGen) {
    isl_ast_node_free(PPCGGen->tree);
    isl_union_map_free(PPCGGen->sizes);
    isl_union_map_free(PPCGGen->used_sizes);
    free(PPCGGen);
  }

  bool runOnScop(Scop &CurrentScop) override {
    S = &CurrentScop;

    auto PPCGScop = createPPCGScop();
    auto PPCGProg = createPPCGProg(PPCGScop);
    auto PPCGGen = generateGPU(PPCGScop, PPCGProg);
    freePPCGGen(PPCGGen);
    gpu_prog_free(PPCGProg);
    ppcg_scop_free(PPCGScop);

    return true;
  }

  void printScop(raw_ostream &, Scop &) const override {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<RegionInfoPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<ScopDetection>();
    AU.addRequired<ScopInfoRegionPass>();
    AU.addRequired<LoopInfoWrapperPass>();

    AU.addPreserved<AAResultsWrapperPass>();
    AU.addPreserved<BasicAAWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.addPreserved<PostDominatorTreeWrapperPass>();
    AU.addPreserved<ScopDetection>();
    AU.addPreserved<ScalarEvolutionWrapperPass>();
    AU.addPreserved<SCEVAAWrapperPass>();

    // FIXME: We do not yet add regions for the newly generated code to the
    //        region tree.
    AU.addPreserved<RegionInfoPass>();
    AU.addPreserved<ScopInfoRegionPass>();
  }
};
}

char PPCGCodeGeneration::ID = 1;

Pass *polly::createPPCGCodeGenerationPass() { return new PPCGCodeGeneration(); }

INITIALIZE_PASS_BEGIN(PPCGCodeGeneration, "polly-codegen-ppcg",
                      "Polly - Apply PPCG translation to SCOP", false, false)
INITIALIZE_PASS_DEPENDENCY(DependenceInfo);
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass);
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass);
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass);
INITIALIZE_PASS_DEPENDENCY(ScopDetection);
INITIALIZE_PASS_END(PPCGCodeGeneration, "polly-codegen-ppcg",
                    "Polly - Apply PPCG translation to SCOP", false, false)
