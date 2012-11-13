//===- Cloog.cpp - Cloog interface ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Cloog[1] interface.
//
// The Cloog interface takes a Scop and generates a Cloog AST (clast). This
// clast can either be returned directly or it can be pretty printed to stdout.
//
// A typical clast output looks like this:
//
// for (c2 = max(0, ceild(n + m, 2); c2 <= min(511, floord(5 * n, 3)); c2++) {
//   bb2(c2);
// }
//
// [1] http://www.cloog.org/ - The Chunky Loop Generator
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/Cloog.h"
#ifdef CLOOG_FOUND
#include "polly/LinkAllPasses.h"
#include "polly/ScopInfo.h"

#define DEBUG_TYPE "polly-cloog"
#include "llvm/Assembly/Writer.h"
#include "llvm/Module.h"
#include "llvm/Support/Debug.h"

#include "cloog/isl/domain.h"
#include "cloog/isl/cloog.h"

#include <unistd.h>

using namespace llvm;
using namespace polly;

namespace polly {
class Cloog {
  Scop *S;
  CloogOptions *Options;
  CloogState *State;
  clast_stmt *ClastRoot;

  void buildCloogOptions();
  CloogUnionDomain *buildCloogUnionDomain();
  CloogInput *buildCloogInput();

public:
  Cloog(Scop *Scop);

  ~Cloog();

  /// Write a .cloog input file
  void dump(FILE *F);

  /// Print a source code representation of the program.
  void pprint(llvm::raw_ostream &OS);

  /// Create the Cloog AST from this program.
  struct clast_root *getClast();
};

Cloog::Cloog(Scop *Scop) : S(Scop) {
  State = cloog_isl_state_malloc(Scop->getIslCtx());
  buildCloogOptions();
  ClastRoot = cloog_clast_create_from_input(buildCloogInput(), Options);
}

Cloog::~Cloog() {
  cloog_clast_free(ClastRoot);
  cloog_options_free(Options);
  cloog_state_free(State);
}

// Create a FILE* write stream and get the output to it written
// to a std::string.
class FileToString {
  int FD[2];
  FILE *input;
  static const int BUFFERSIZE = 20;

  char buf[BUFFERSIZE + 1];


public:
  FileToString() {
    pipe(FD);
    input = fdopen(FD[1], "w");
  }
  ~FileToString() {
    close(FD[0]);
    //close(FD[1]);
  }

  FILE *getInputFile() {
    return input;
  }

  void closeInput() {
    fclose(input);
    close(FD[1]);
  }

  std::string getOutput() {
    std::string output;
    int readSize;

    while (true) {
      readSize = read(FD[0], &buf, BUFFERSIZE);

      if (readSize <= 0)
        break;

      output += std::string(buf, readSize);
    }


    return output;
  }

};

/// Write .cloog input file.
void Cloog::dump(FILE *F) {
  CloogInput *Input = buildCloogInput();
  cloog_input_dump_cloog(F, Input, Options);
  cloog_input_free(Input);
}

/// Print a source code representation of the program.
void Cloog::pprint(raw_ostream &OS) {
  FileToString *Output = new FileToString();
  clast_pprint(Output->getInputFile(), ClastRoot, 0, Options);
  Output->closeInput();
  OS << Output->getOutput();
  delete (Output);
}

/// Create the Cloog AST from this program.
struct clast_root *Cloog::getClast() {
  return (clast_root*)ClastRoot;
}

void Cloog::buildCloogOptions() {
  Options = cloog_options_malloc(State);
  Options->quiet = 1;
  Options->strides = 1;
  Options->save_domains = 1;
  Options->noscalars = 1;

  // The last loop depth to optimize should be the last scattering dimension.
  // CLooG by default will continue to split the loops even after the last
  // scattering dimension. This splitting is problematic for the schedules
  // calculated by the PoCC/isl/Pluto optimizer. Such schedules contain may
  // not be fully defined, but statements without dependences may be mapped
  // to the same exeuction time. For such schedules, continuing to split
  // may lead to a larger set of if-conditions in the innermost loop.
  Options->l = 0;
}

CloogUnionDomain *Cloog::buildCloogUnionDomain() {
  CloogUnionDomain *DU = cloog_union_domain_alloc(S->getNumParams());

  for (Scop::iterator SI = S->begin(), SE = S->end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;
    CloogScattering *Scattering;
    CloogDomain *Domain;

    Scattering = cloog_scattering_from_isl_map(Stmt->getScattering());
    Domain  = cloog_domain_from_isl_set(Stmt->getDomain());

    std::string entryName = Stmt->getBaseName();

    DU = cloog_union_domain_add_domain(DU, entryName.c_str(), Domain,
                                       Scattering, Stmt);
  }

  return DU;
}

CloogInput *Cloog::buildCloogInput() {
  // XXX: We do not copy the context of the scop, but use an unconstrained
  //      context. This 'hack' is necessary as the context may contain bounds
  //      on parameters such as [n] -> {:0 <= n < 2^32}. Those large
  //      integers will cause CLooG to construct a clast that contains
  //      expressions that include these large integers. Such expressions can
  //      possibly not be evaluated correctly with i64 types. The cloog
  //      based code generation backend, however, can not derive types
  //      automatically and just assumes i64 types. Hence, it will break or
  //      generate incorrect code.
  //      This hack does not remove all possibilities of incorrectly generated
  //      code, but it is ensures that for most problems the problems do not
  //      show up. The correct solution, will be to automatically derive the
  //      minimal types for each expression. This could be added to CLooG and it
  //      will be available in the isl based code generation.
  isl_set *EmptyContext = isl_set_universe(S->getParamSpace());
  CloogDomain *Context = cloog_domain_from_isl_set(EmptyContext);
  CloogUnionDomain *Statements = buildCloogUnionDomain();

  isl_set *ScopContext = S->getContext();

  for (unsigned i = 0; i < isl_set_dim(ScopContext, isl_dim_param); i++) {
    isl_id *id = isl_set_get_dim_id(ScopContext, isl_dim_param, i);
    Statements = cloog_union_domain_set_name(Statements, CLOOG_PARAM, i,
                                             isl_id_get_name(id));
    isl_id_free(id);
  }

  isl_set_free(ScopContext);

  CloogInput *Input = cloog_input_alloc(Context, Statements);
  return Input;
}

void ClastVisitor::visit(const clast_stmt *stmt) {
  if      (CLAST_STMT_IS_A(stmt, stmt_root))
    assert(false && "No second root statement expected");
  else if (CLAST_STMT_IS_A(stmt, stmt_ass))
    return visitAssignment((const clast_assignment *)stmt);
  else if (CLAST_STMT_IS_A(stmt, stmt_user))
    return visitUser((const clast_user_stmt *)stmt);
  else if (CLAST_STMT_IS_A(stmt, stmt_block))
    return visitBlock((const clast_block *)stmt);
  else if (CLAST_STMT_IS_A(stmt, stmt_for))
    return visitFor((const clast_for *)stmt);
  else if (CLAST_STMT_IS_A(stmt, stmt_guard))
    return visitGuard((const clast_guard *)stmt);

  if (stmt->next)
    visit(stmt->next);
}

void ClastVisitor::visitAssignment(const clast_assignment *stmt) {
}

void ClastVisitor::visitBlock(const clast_block *stmt) {
  visit(stmt->body);
}

void ClastVisitor::visitFor(const clast_for *stmt) {
  visit(stmt->body);
}

void ClastVisitor::visitGuard(const clast_guard *stmt) {
  visit(stmt->then);
}

} // End namespace polly.

namespace {

struct CloogExporter : public ScopPass {
  static char ID;
  Scop *S;
  explicit CloogExporter() : ScopPass(ID) {}

  std::string getFileName(Region *R) const;
  virtual bool runOnScop(Scop &S);
  void getAnalysisUsage(AnalysisUsage &AU) const;
};

}
std::string CloogExporter::getFileName(Region *R) const {
  std::string FunctionName = R->getEntry()->getParent()->getName();
  std::string ExitName, EntryName;

  raw_string_ostream ExitStr(ExitName);
  raw_string_ostream EntryStr(EntryName);

  WriteAsOperand(EntryStr, R->getEntry(), false);
  EntryStr.str();

  if (R->getExit()) {
    WriteAsOperand(ExitStr, R->getExit(), false);
    ExitStr.str();
  } else
    ExitName = "FunctionExit";

  std::string RegionName = EntryName + "---" + ExitName;
  std::string FileName = FunctionName + "___" + RegionName + ".cloog";

  return FileName;
}

char CloogExporter::ID = 0;
bool CloogExporter::runOnScop(Scop &S) {
  Region &R = S.getRegion();
  CloogInfo &C = getAnalysis<CloogInfo>();

  std::string FunctionName = R.getEntry()->getParent()->getName();
  std::string Filename = getFileName(&R);

  errs() << "Writing Scop '" << R.getNameStr() << "' in function '"
    << FunctionName << "' to '" << Filename << "'...\n";

  FILE *F = fopen(Filename.c_str(), "w");
  C.dump(F);
  fclose(F);

  return false;
}

void CloogExporter::getAnalysisUsage(AnalysisUsage &AU) const {
  // Get the Common analysis usage of ScopPasses.
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<CloogInfo>();
}

static RegisterPass<CloogExporter> A("polly-export-cloog",
                                    "Polly - Export the Cloog input file"
                                    " (Writes a .cloog file for each Scop)"
                                    );

llvm::Pass *polly::createCloogExporterPass() {
  return new CloogExporter();
}

/// Write a .cloog input file
void CloogInfo::dump(FILE *F) {
  C->dump(F);
}

/// Print a source code representation of the program.
void CloogInfo::pprint(llvm::raw_ostream &OS) {
  C->pprint(OS);
}

/// Create the Cloog AST from this program.
const struct clast_root *CloogInfo::getClast() {
  return C->getClast();
}

void CloogInfo::releaseMemory() {
  if (C) {
    delete C;
    C = 0;
  }
}

bool CloogInfo::runOnScop(Scop &S) {
  if (C)
    delete C;

  scop = &S;

  C = new Cloog(&S);

  Function *F = S.getRegion().getEntry()->getParent();

  DEBUG(dbgs() << ":: " << F->getName());
  DEBUG(dbgs() << " : " << S.getRegion().getNameStr() << "\n");;
  DEBUG(C->pprint(dbgs()));

  return false;
}

void CloogInfo::printScop(raw_ostream &OS) const {
  Function *function = scop->getRegion().getEntry()->getParent();

  OS << function->getName() << "():\n";

  C->pprint(OS);
}

void CloogInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  // Get the Common analysis usage of ScopPasses.
  ScopPass::getAnalysisUsage(AU);
}
char CloogInfo::ID = 0;

INITIALIZE_PASS_BEGIN(CloogInfo, "polly-cloog",
                      "Execute Cloog code generation", false, false)
INITIALIZE_PASS_DEPENDENCY(ScopInfo)
INITIALIZE_PASS_END(CloogInfo, "polly-cloog",
                    "Execute Cloog code generation", false, false)

Pass *polly::createCloogInfoPass() {
  return new CloogInfo();
}

#endif // CLOOG_FOUND
