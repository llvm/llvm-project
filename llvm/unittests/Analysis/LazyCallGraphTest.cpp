//===- LazyCallGraphTest.cpp - Unit tests for the lazy CG analysis --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;

namespace {

std::unique_ptr<Module> parseAssembly(LLVMContext &Context,
                                      const char *Assembly) {
  SMDiagnostic Error;
  std::unique_ptr<Module> M = parseAssemblyString(Assembly, Error, Context);

  std::string ErrMsg;
  raw_string_ostream OS(ErrMsg);
  Error.print("", OS);

  // A failure here means that the test itself is buggy.
  if (!M)
    report_fatal_error(OS.str().c_str());

  return M;
}

/*
   IR forming a call graph with a diamond of triangle-shaped SCCs:

           d1
          /  \
         d3--d2
        /     \
       b1     c1
     /  \    /  \
    b3--b2  c3--c2
         \  /
          a1
         /  \
        a3--a2

   All call edges go up between SCCs, and clockwise around the SCC.
 */
static const char DiamondOfTriangles[] =
     "define void @a1() {\n"
     "entry:\n"
     "  call void @a2()\n"
     "  call void @b2()\n"
     "  call void @c3()\n"
     "  ret void\n"
     "}\n"
     "define void @a2() {\n"
     "entry:\n"
     "  call void @a3()\n"
     "  ret void\n"
     "}\n"
     "define void @a3() {\n"
     "entry:\n"
     "  call void @a1()\n"
     "  ret void\n"
     "}\n"
     "define void @b1() {\n"
     "entry:\n"
     "  call void @b2()\n"
     "  call void @d3()\n"
     "  ret void\n"
     "}\n"
     "define void @b2() {\n"
     "entry:\n"
     "  call void @b3()\n"
     "  ret void\n"
     "}\n"
     "define void @b3() {\n"
     "entry:\n"
     "  call void @b1()\n"
     "  ret void\n"
     "}\n"
     "define void @c1() {\n"
     "entry:\n"
     "  call void @c2()\n"
     "  call void @d2()\n"
     "  ret void\n"
     "}\n"
     "define void @c2() {\n"
     "entry:\n"
     "  call void @c3()\n"
     "  ret void\n"
     "}\n"
     "define void @c3() {\n"
     "entry:\n"
     "  call void @c1()\n"
     "  ret void\n"
     "}\n"
     "define void @d1() {\n"
     "entry:\n"
     "  call void @d2()\n"
     "  ret void\n"
     "}\n"
     "define void @d2() {\n"
     "entry:\n"
     "  call void @d3()\n"
     "  ret void\n"
     "}\n"
     "define void @d3() {\n"
     "entry:\n"
     "  call void @d1()\n"
     "  ret void\n"
     "}\n";

/*
   IR forming a reference graph with a diamond of triangle-shaped RefSCCs

           d1
          /  \
         d3--d2
        /     \
       b1     c1
     /  \    /  \
    b3--b2  c3--c2
         \  /
          a1
         /  \
        a3--a2

   All call edges go up between RefSCCs, and clockwise around the RefSCC.
 */
static const char DiamondOfTrianglesRefGraph[] =
     "define void @a1() {\n"
     "entry:\n"
     "  %a = alloca void ()*\n"
     "  store void ()* @a2, void ()** %a\n"
     "  store void ()* @b2, void ()** %a\n"
     "  store void ()* @c3, void ()** %a\n"
     "  ret void\n"
     "}\n"
     "define void @a2() {\n"
     "entry:\n"
     "  %a = alloca void ()*\n"
     "  store void ()* @a3, void ()** %a\n"
     "  ret void\n"
     "}\n"
     "define void @a3() {\n"
     "entry:\n"
     "  %a = alloca void ()*\n"
     "  store void ()* @a1, void ()** %a\n"
     "  ret void\n"
     "}\n"
     "define void @b1() {\n"
     "entry:\n"
     "  %a = alloca void ()*\n"
     "  store void ()* @b2, void ()** %a\n"
     "  store void ()* @d3, void ()** %a\n"
     "  ret void\n"
     "}\n"
     "define void @b2() {\n"
     "entry:\n"
     "  %a = alloca void ()*\n"
     "  store void ()* @b3, void ()** %a\n"
     "  ret void\n"
     "}\n"
     "define void @b3() {\n"
     "entry:\n"
     "  %a = alloca void ()*\n"
     "  store void ()* @b1, void ()** %a\n"
     "  ret void\n"
     "}\n"
     "define void @c1() {\n"
     "entry:\n"
     "  %a = alloca void ()*\n"
     "  store void ()* @c2, void ()** %a\n"
     "  store void ()* @d2, void ()** %a\n"
     "  ret void\n"
     "}\n"
     "define void @c2() {\n"
     "entry:\n"
     "  %a = alloca void ()*\n"
     "  store void ()* @c3, void ()** %a\n"
     "  ret void\n"
     "}\n"
     "define void @c3() {\n"
     "entry:\n"
     "  %a = alloca void ()*\n"
     "  store void ()* @c1, void ()** %a\n"
     "  ret void\n"
     "}\n"
     "define void @d1() {\n"
     "entry:\n"
     "  %a = alloca void ()*\n"
     "  store void ()* @d2, void ()** %a\n"
     "  ret void\n"
     "}\n"
     "define void @d2() {\n"
     "entry:\n"
     "  %a = alloca void ()*\n"
     "  store void ()* @d3, void ()** %a\n"
     "  ret void\n"
     "}\n"
     "define void @d3() {\n"
     "entry:\n"
     "  %a = alloca void ()*\n"
     "  store void ()* @d1, void ()** %a\n"
     "  ret void\n"
     "}\n";

static LazyCallGraph buildCG(Module &M) {
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
  TargetLibraryInfo TLI(TLII);
  LazyCallGraph CG(M, TLI);
  return CG;
}

TEST(LazyCallGraphTest, BasicGraphFormation) {
  LLVMContext Context;
  std::unique_ptr<Module> M = parseAssembly(Context, DiamondOfTriangles);
  LazyCallGraph CG = buildCG(*M);

  // The order of the entry nodes should be stable w.r.t. the source order of
  // the IR, and everything in our module is an entry node, so just directly
  // build variables for each node.
  auto I = CG.begin();
  LazyCallGraph::Node &A1 = (I++)->getNode();
  EXPECT_EQ("a1", A1.getFunction().getName());
  LazyCallGraph::Node &A2 = (I++)->getNode();
  EXPECT_EQ("a2", A2.getFunction().getName());
  LazyCallGraph::Node &A3 = (I++)->getNode();
  EXPECT_EQ("a3", A3.getFunction().getName());
  LazyCallGraph::Node &B1 = (I++)->getNode();
  EXPECT_EQ("b1", B1.getFunction().getName());
  LazyCallGraph::Node &B2 = (I++)->getNode();
  EXPECT_EQ("b2", B2.getFunction().getName());
  LazyCallGraph::Node &B3 = (I++)->getNode();
  EXPECT_EQ("b3", B3.getFunction().getName());
  LazyCallGraph::Node &C1 = (I++)->getNode();
  EXPECT_EQ("c1", C1.getFunction().getName());
  LazyCallGraph::Node &C2 = (I++)->getNode();
  EXPECT_EQ("c2", C2.getFunction().getName());
  LazyCallGraph::Node &C3 = (I++)->getNode();
  EXPECT_EQ("c3", C3.getFunction().getName());
  LazyCallGraph::Node &D1 = (I++)->getNode();
  EXPECT_EQ("d1", D1.getFunction().getName());
  LazyCallGraph::Node &D2 = (I++)->getNode();
  EXPECT_EQ("d2", D2.getFunction().getName());
  LazyCallGraph::Node &D3 = (I++)->getNode();
  EXPECT_EQ("d3", D3.getFunction().getName());
  EXPECT_EQ(CG.end(), I);

  // Build vectors and sort them for the rest of the assertions to make them
  // independent of order.
  std::vector<std::string> Nodes;

  for (LazyCallGraph::Edge &E : A1.populate())
    Nodes.push_back(E.getFunction().getName());
  std::sort(Nodes.begin(), Nodes.end());
  EXPECT_EQ("a2", Nodes[0]);
  EXPECT_EQ("b2", Nodes[1]);
  EXPECT_EQ("c3", Nodes[2]);
  Nodes.clear();

  A2.populate();
  EXPECT_EQ(A2->end(), std::next(A2->begin()));
  EXPECT_EQ("a3", A2->begin()->getFunction().getName());
  A3.populate();
  EXPECT_EQ(A3->end(), std::next(A3->begin()));
  EXPECT_EQ("a1", A3->begin()->getFunction().getName());

  for (LazyCallGraph::Edge &E : B1.populate())
    Nodes.push_back(E.getFunction().getName());
  std::sort(Nodes.begin(), Nodes.end());
  EXPECT_EQ("b2", Nodes[0]);
  EXPECT_EQ("d3", Nodes[1]);
  Nodes.clear();

  B2.populate();
  EXPECT_EQ(B2->end(), std::next(B2->begin()));
  EXPECT_EQ("b3", B2->begin()->getFunction().getName());
  B3.populate();
  EXPECT_EQ(B3->end(), std::next(B3->begin()));
  EXPECT_EQ("b1", B3->begin()->getFunction().getName());

  for (LazyCallGraph::Edge &E : C1.populate())
    Nodes.push_back(E.getFunction().getName());
  std::sort(Nodes.begin(), Nodes.end());
  EXPECT_EQ("c2", Nodes[0]);
  EXPECT_EQ("d2", Nodes[1]);
  Nodes.clear();

  C2.populate();
  EXPECT_EQ(C2->end(), std::next(C2->begin()));
  EXPECT_EQ("c3", C2->begin()->getFunction().getName());
  C3.populate();
  EXPECT_EQ(C3->end(), std::next(C3->begin()));
  EXPECT_EQ("c1", C3->begin()->getFunction().getName());

  D1.populate();
  EXPECT_EQ(D1->end(), std::next(D1->begin()));
  EXPECT_EQ("d2", D1->begin()->getFunction().getName());
  D2.populate();
  EXPECT_EQ(D2->end(), std::next(D2->begin()));
  EXPECT_EQ("d3", D2->begin()->getFunction().getName());
  D3.populate();
  EXPECT_EQ(D3->end(), std::next(D3->begin()));
  EXPECT_EQ("d1", D3->begin()->getFunction().getName());

  // Now lets look at the RefSCCs and SCCs.
  CG.buildRefSCCs();
  auto J = CG.postorder_ref_scc_begin();

  LazyCallGraph::RefSCC &D = *J++;
  ASSERT_EQ(1, D.size());
  for (LazyCallGraph::Node &N : *D.begin())
    Nodes.push_back(N.getFunction().getName());
  std::sort(Nodes.begin(), Nodes.end());
  EXPECT_EQ(3u, Nodes.size());
  EXPECT_EQ("d1", Nodes[0]);
  EXPECT_EQ("d2", Nodes[1]);
  EXPECT_EQ("d3", Nodes[2]);
  Nodes.clear();
  EXPECT_FALSE(D.isParentOf(D));
  EXPECT_FALSE(D.isChildOf(D));
  EXPECT_FALSE(D.isAncestorOf(D));
  EXPECT_FALSE(D.isDescendantOf(D));
  EXPECT_EQ(&D, &*CG.postorder_ref_scc_begin());

  LazyCallGraph::RefSCC &C = *J++;
  ASSERT_EQ(1, C.size());
  for (LazyCallGraph::Node &N : *C.begin())
    Nodes.push_back(N.getFunction().getName());
  std::sort(Nodes.begin(), Nodes.end());
  EXPECT_EQ(3u, Nodes.size());
  EXPECT_EQ("c1", Nodes[0]);
  EXPECT_EQ("c2", Nodes[1]);
  EXPECT_EQ("c3", Nodes[2]);
  Nodes.clear();
  EXPECT_TRUE(C.isParentOf(D));
  EXPECT_FALSE(C.isChildOf(D));
  EXPECT_TRUE(C.isAncestorOf(D));
  EXPECT_FALSE(C.isDescendantOf(D));
  EXPECT_EQ(&C, &*std::next(CG.postorder_ref_scc_begin()));

  LazyCallGraph::RefSCC &B = *J++;
  ASSERT_EQ(1, B.size());
  for (LazyCallGraph::Node &N : *B.begin())
    Nodes.push_back(N.getFunction().getName());
  std::sort(Nodes.begin(), Nodes.end());
  EXPECT_EQ(3u, Nodes.size());
  EXPECT_EQ("b1", Nodes[0]);
  EXPECT_EQ("b2", Nodes[1]);
  EXPECT_EQ("b3", Nodes[2]);
  Nodes.clear();
  EXPECT_TRUE(B.isParentOf(D));
  EXPECT_FALSE(B.isChildOf(D));
  EXPECT_TRUE(B.isAncestorOf(D));
  EXPECT_FALSE(B.isDescendantOf(D));
  EXPECT_FALSE(B.isAncestorOf(C));
  EXPECT_FALSE(C.isAncestorOf(B));
  EXPECT_EQ(&B, &*std::next(CG.postorder_ref_scc_begin(), 2));

  LazyCallGraph::RefSCC &A = *J++;
  ASSERT_EQ(1, A.size());
  for (LazyCallGraph::Node &N : *A.begin())
    Nodes.push_back(N.getFunction().getName());
  std::sort(Nodes.begin(), Nodes.end());
  EXPECT_EQ(3u, Nodes.size());
  EXPECT_EQ("a1", Nodes[0]);
  EXPECT_EQ("a2", Nodes[1]);
  EXPECT_EQ("a3", Nodes[2]);
  Nodes.clear();
  EXPECT_TRUE(A.isParentOf(B));
  EXPECT_TRUE(A.isParentOf(C));
  EXPECT_FALSE(A.isParentOf(D));
  EXPECT_TRUE(A.isAncestorOf(B));
  EXPECT_TRUE(A.isAncestorOf(C));
  EXPECT_TRUE(A.isAncestorOf(D));
  EXPECT_EQ(&A, &*std::next(CG.postorder_ref_scc_begin(), 3));

  EXPECT_EQ(CG.postorder_ref_scc_end(), J);
  EXPECT_EQ(J, std::next(CG.postorder_ref_scc_begin(), 4));
}

static Function &lookupFunction(Module &M, StringRef Name) {
  for (Function &F : M)
    if (F.getName() == Name)
      return F;
  report_fatal_error("Couldn't find function!");
}

TEST(LazyCallGraphTest, BasicGraphMutation) {
  LLVMContext Context;
  std::unique_ptr<Module> M = parseAssembly(Context, "define void @a() {\n"
                                                     "entry:\n"
                                                     "  call void @b()\n"
                                                     "  call void @c()\n"
                                                     "  ret void\n"
                                                     "}\n"
                                                     "define void @b() {\n"
                                                     "entry:\n"
                                                     "  ret void\n"
                                                     "}\n"
                                                     "define void @c() {\n"
                                                     "entry:\n"
                                                     "  ret void\n"
                                                     "}\n");
  LazyCallGraph CG = buildCG(*M);

  LazyCallGraph::Node &A = CG.get(lookupFunction(*M, "a"));
  LazyCallGraph::Node &B = CG.get(lookupFunction(*M, "b"));
  A.populate();
  EXPECT_EQ(2, std::distance(A->begin(), A->end()));
  B.populate();
  EXPECT_EQ(0, std::distance(B->begin(), B->end()));

  LazyCallGraph::Node &C = CG.get(lookupFunction(*M, "c"));
  C.populate();
  CG.insertEdge(B, C, LazyCallGraph::Edge::Call);
  EXPECT_EQ(1, std::distance(B->begin(), B->end()));
  EXPECT_EQ(0, std::distance(C->begin(), C->end()));

  CG.insertEdge(C, B, LazyCallGraph::Edge::Call);
  EXPECT_EQ(1, std::distance(C->begin(), C->end()));
  EXPECT_EQ(&B, &C->begin()->getNode());

  CG.insertEdge(C, C, LazyCallGraph::Edge::Call);
  EXPECT_EQ(2, std::distance(C->begin(), C->end()));
  EXPECT_EQ(&B, &C->begin()->getNode());
  EXPECT_EQ(&C, &std::next(C->begin())->getNode());

  CG.removeEdge(C, B);
  EXPECT_EQ(1, std::distance(C->begin(), C->end()));
  EXPECT_EQ(&C, &C->begin()->getNode());

  CG.removeEdge(C, C);
  EXPECT_EQ(0, std::distance(C->begin(), C->end()));

  CG.removeEdge(B, C);
  EXPECT_EQ(0, std::distance(B->begin(), B->end()));
}

TEST(LazyCallGraphTest, InnerSCCFormation) {
  LLVMContext Context;
  std::unique_ptr<Module> M = parseAssembly(Context, DiamondOfTriangles);
  LazyCallGraph CG = buildCG(*M);

  // Now mutate the graph to connect every node into a single RefSCC to ensure
  // that our inner SCC formation handles the rest.
  LazyCallGraph::Node &D1 = CG.get(lookupFunction(*M, "d1"));
  LazyCallGraph::Node &A1 = CG.get(lookupFunction(*M, "a1"));
  A1.populate();
  D1.populate();
  CG.insertEdge(D1, A1, LazyCallGraph::Edge::Ref);

  // Build vectors and sort them for the rest of the assertions to make them
  // independent of order.
  std::vector<std::string> Nodes;

  // We should build a single RefSCC for the entire graph.
  CG.buildRefSCCs();
  auto I = CG.postorder_ref_scc_begin();
  LazyCallGraph::RefSCC &RC = *I++;
  EXPECT_EQ(CG.postorder_ref_scc_end(), I);

  // Now walk the four SCCs which should be in post-order.
  auto J = RC.begin();
  LazyCallGraph::SCC &D = *J++;
  for (LazyCallGraph::Node &N : D)
    Nodes.push_back(N.getFunction().getName());
  std::sort(Nodes.begin(), Nodes.end());
  EXPECT_EQ(3u, Nodes.size());
  EXPECT_EQ("d1", Nodes[0]);
  EXPECT_EQ("d2", Nodes[1]);
  EXPECT_EQ("d3", Nodes[2]);
  Nodes.clear();

  LazyCallGraph::SCC &B = *J++;
  for (LazyCallGraph::Node &N : B)
    Nodes.push_back(N.getFunction().getName());
  std::sort(Nodes.begin(), Nodes.end());
  EXPECT_EQ(3u, Nodes.size());
  EXPECT_EQ("b1", Nodes[0]);
  EXPECT_EQ("b2", Nodes[1]);
  EXPECT_EQ("b3", Nodes[2]);
  Nodes.clear();

  LazyCallGraph::SCC &C = *J++;
  for (LazyCallGraph::Node &N : C)
    Nodes.push_back(N.getFunction().getName());
  std::sort(Nodes.begin(), Nodes.end());
  EXPECT_EQ(3u, Nodes.size());
  EXPECT_EQ("c1", Nodes[0]);
  EXPECT_EQ("c2", Nodes[1]);
  EXPECT_EQ("c3", Nodes[2]);
  Nodes.clear();

  LazyCallGraph::SCC &A = *J++;
  for (LazyCallGraph::Node &N : A)
    Nodes.push_back(N.getFunction().getName());
  std::sort(Nodes.begin(), Nodes.end());
  EXPECT_EQ(3u, Nodes.size());
  EXPECT_EQ("a1", Nodes[0]);
  EXPECT_EQ("a2", Nodes[1]);
  EXPECT_EQ("a3", Nodes[2]);
  Nodes.clear();

  EXPECT_EQ(RC.end(), J);
}

TEST(LazyCallGraphTest, MultiArmSCC) {
  LLVMContext Context;
  // Two interlocking cycles. The really useful thing about this SCC is that it
  // will require Tarjan's DFS to backtrack and finish processing all of the
  // children of each node in the SCC. Since this involves call edges, both
  // Tarjan implementations will have to successfully navigate the structure.
  std::unique_ptr<Module> M = parseAssembly(Context, "define void @f1() {\n"
                                                     "entry:\n"
                                                     "  call void @f2()\n"
                                                     "  call void @f4()\n"
                                                     "  ret void\n"
                                                     "}\n"
                                                     "define void @f2() {\n"
                                                     "entry:\n"
                                                     "  call void @f3()\n"
                                                     "  ret void\n"
                                                     "}\n"
                                                     "define void @f3() {\n"
                                                     "entry:\n"
                                                     "  call void @f1()\n"
                                                     "  ret void\n"
                                                     "}\n"
                                                     "define void @f4() {\n"
                                                     "entry:\n"
                                                     "  call void @f5()\n"
                                                     "  ret void\n"
                                                     "}\n"
                                                     "define void @f5() {\n"
                                                     "entry:\n"
                                                     "  call void @f1()\n"
                                                     "  ret void\n"
                                                     "}\n");
  LazyCallGraph CG = buildCG(*M);

  // Force the graph to be fully expanded.
  CG.buildRefSCCs();
  auto I = CG.postorder_ref_scc_begin();
  LazyCallGraph::RefSCC &RC = *I++;
  EXPECT_EQ(CG.postorder_ref_scc_end(), I);

  LazyCallGraph::Node &N1 = *CG.lookup(lookupFunction(*M, "f1"));
  LazyCallGraph::Node &N2 = *CG.lookup(lookupFunction(*M, "f2"));
  LazyCallGraph::Node &N3 = *CG.lookup(lookupFunction(*M, "f3"));
  LazyCallGraph::Node &N4 = *CG.lookup(lookupFunction(*M, "f4"));
  LazyCallGraph::Node &N5 = *CG.lookup(lookupFunction(*M, "f4"));
  EXPECT_EQ(&RC, CG.lookupRefSCC(N1));
  EXPECT_EQ(&RC, CG.lookupRefSCC(N2));
  EXPECT_EQ(&RC, CG.lookupRefSCC(N3));
  EXPECT_EQ(&RC, CG.lookupRefSCC(N4));
  EXPECT_EQ(&RC, CG.lookupRefSCC(N5));

  ASSERT_EQ(1, RC.size());

  LazyCallGraph::SCC &C = *RC.begin();
  EXPECT_EQ(&C, CG.lookupSCC(N1));
  EXPECT_EQ(&C, CG.lookupSCC(N2));
  EXPECT_EQ(&C, CG.lookupSCC(N3));
  EXPECT_EQ(&C, CG.lookupSCC(N4));
  EXPECT_EQ(&C, CG.lookupSCC(N5));
}

TEST(LazyCallGraphTest, OutgoingEdgeMutation) {
  LLVMContext Context;
  std::unique_ptr<Module> M = parseAssembly(Context, "define void @a() {\n"
                                                     "entry:\n"
                                                     "  call void @b()\n"
                                                     "  call void @c()\n"
                                                     "  ret void\n"
                                                     "}\n"
                                                     "define void @b() {\n"
                                                     "entry:\n"
                                                     "  call void @d()\n"
                                                     "  ret void\n"
                                                     "}\n"
                                                     "define void @c() {\n"
                                                     "entry:\n"
                                                     "  call void @d()\n"
                                                     "  ret void\n"
                                                     "}\n"
                                                     "define void @d() {\n"
                                                     "entry:\n"
                                                     "  ret void\n"
                                                     "}\n");
  LazyCallGraph CG = buildCG(*M);

  // Force the graph to be fully expanded.
  CG.buildRefSCCs();
  for (LazyCallGraph::RefSCC &RC : CG.postorder_ref_sccs())
    dbgs() << "Formed RefSCC: " << RC << "\n";

  LazyCallGraph::Node &A = *CG.lookup(lookupFunction(*M, "a"));
  LazyCallGraph::Node &B = *CG.lookup(lookupFunction(*M, "b"));
  LazyCallGraph::Node &C = *CG.lookup(lookupFunction(*M, "c"));
  LazyCallGraph::Node &D = *CG.lookup(lookupFunction(*M, "d"));
  LazyCallGraph::SCC &AC = *CG.lookupSCC(A);
  LazyCallGraph::SCC &BC = *CG.lookupSCC(B);
  LazyCallGraph::SCC &CC = *CG.lookupSCC(C);
  LazyCallGraph::SCC &DC = *CG.lookupSCC(D);
  LazyCallGraph::RefSCC &ARC = *CG.lookupRefSCC(A);
  LazyCallGraph::RefSCC &BRC = *CG.lookupRefSCC(B);
  LazyCallGraph::RefSCC &CRC = *CG.lookupRefSCC(C);
  LazyCallGraph::RefSCC &DRC = *CG.lookupRefSCC(D);
  EXPECT_TRUE(ARC.isParentOf(BRC));
  EXPECT_TRUE(AC.isParentOf(BC));
  EXPECT_TRUE(ARC.isParentOf(CRC));
  EXPECT_TRUE(AC.isParentOf(CC));
  EXPECT_FALSE(ARC.isParentOf(DRC));
  EXPECT_FALSE(AC.isParentOf(DC));
  EXPECT_TRUE(ARC.isAncestorOf(DRC));
  EXPECT_TRUE(AC.isAncestorOf(DC));
  EXPECT_FALSE(DRC.isChildOf(ARC));
  EXPECT_FALSE(DC.isChildOf(AC));
  EXPECT_TRUE(DRC.isDescendantOf(ARC));
  EXPECT_TRUE(DC.isDescendantOf(AC));
  EXPECT_TRUE(DRC.isChildOf(BRC));
  EXPECT_TRUE(DC.isChildOf(BC));
  EXPECT_TRUE(DRC.isChildOf(CRC));
  EXPECT_TRUE(DC.isChildOf(CC));

  EXPECT_EQ(2, std::distance(A->begin(), A->end()));
  ARC.insertOutgoingEdge(A, D, LazyCallGraph::Edge::Call);
  EXPECT_EQ(3, std::distance(A->begin(), A->end()));
  const LazyCallGraph::Edge &NewE = (*A)[D];
  EXPECT_TRUE(NewE);
  EXPECT_TRUE(NewE.isCall());
  EXPECT_EQ(&D, &NewE.getNode());

  // Only the parent and child tests sholud have changed. The rest of the graph
  // remains the same.
  EXPECT_TRUE(ARC.isParentOf(DRC));
  EXPECT_TRUE(AC.isParentOf(DC));
  EXPECT_TRUE(ARC.isAncestorOf(DRC));
  EXPECT_TRUE(AC.isAncestorOf(DC));
  EXPECT_TRUE(DRC.isChildOf(ARC));
  EXPECT_TRUE(DC.isChildOf(AC));
  EXPECT_TRUE(DRC.isDescendantOf(ARC));
  EXPECT_TRUE(DC.isDescendantOf(AC));
  EXPECT_EQ(&AC, CG.lookupSCC(A));
  EXPECT_EQ(&BC, CG.lookupSCC(B));
  EXPECT_EQ(&CC, CG.lookupSCC(C));
  EXPECT_EQ(&DC, CG.lookupSCC(D));
  EXPECT_EQ(&ARC, CG.lookupRefSCC(A));
  EXPECT_EQ(&BRC, CG.lookupRefSCC(B));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(C));
  EXPECT_EQ(&DRC, CG.lookupRefSCC(D));

  ARC.switchOutgoingEdgeToRef(A, D);
  EXPECT_FALSE(NewE.isCall());

  // Verify the reference graph remains the same but the SCC graph is updated.
  EXPECT_TRUE(ARC.isParentOf(DRC));
  EXPECT_FALSE(AC.isParentOf(DC));
  EXPECT_TRUE(ARC.isAncestorOf(DRC));
  EXPECT_TRUE(AC.isAncestorOf(DC));
  EXPECT_TRUE(DRC.isChildOf(ARC));
  EXPECT_FALSE(DC.isChildOf(AC));
  EXPECT_TRUE(DRC.isDescendantOf(ARC));
  EXPECT_TRUE(DC.isDescendantOf(AC));
  EXPECT_EQ(&AC, CG.lookupSCC(A));
  EXPECT_EQ(&BC, CG.lookupSCC(B));
  EXPECT_EQ(&CC, CG.lookupSCC(C));
  EXPECT_EQ(&DC, CG.lookupSCC(D));
  EXPECT_EQ(&ARC, CG.lookupRefSCC(A));
  EXPECT_EQ(&BRC, CG.lookupRefSCC(B));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(C));
  EXPECT_EQ(&DRC, CG.lookupRefSCC(D));

  ARC.switchOutgoingEdgeToCall(A, D);
  EXPECT_TRUE(NewE.isCall());

  // Verify the reference graph remains the same but the SCC graph is updated.
  EXPECT_TRUE(ARC.isParentOf(DRC));
  EXPECT_TRUE(AC.isParentOf(DC));
  EXPECT_TRUE(ARC.isAncestorOf(DRC));
  EXPECT_TRUE(AC.isAncestorOf(DC));
  EXPECT_TRUE(DRC.isChildOf(ARC));
  EXPECT_TRUE(DC.isChildOf(AC));
  EXPECT_TRUE(DRC.isDescendantOf(ARC));
  EXPECT_TRUE(DC.isDescendantOf(AC));
  EXPECT_EQ(&AC, CG.lookupSCC(A));
  EXPECT_EQ(&BC, CG.lookupSCC(B));
  EXPECT_EQ(&CC, CG.lookupSCC(C));
  EXPECT_EQ(&DC, CG.lookupSCC(D));
  EXPECT_EQ(&ARC, CG.lookupRefSCC(A));
  EXPECT_EQ(&BRC, CG.lookupRefSCC(B));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(C));
  EXPECT_EQ(&DRC, CG.lookupRefSCC(D));

  ARC.removeOutgoingEdge(A, D);
  EXPECT_EQ(2, std::distance(A->begin(), A->end()));

  // Now the parent and child tests fail again but the rest remains the same.
  EXPECT_FALSE(ARC.isParentOf(DRC));
  EXPECT_FALSE(AC.isParentOf(DC));
  EXPECT_TRUE(ARC.isAncestorOf(DRC));
  EXPECT_TRUE(AC.isAncestorOf(DC));
  EXPECT_FALSE(DRC.isChildOf(ARC));
  EXPECT_FALSE(DC.isChildOf(AC));
  EXPECT_TRUE(DRC.isDescendantOf(ARC));
  EXPECT_TRUE(DC.isDescendantOf(AC));
  EXPECT_EQ(&AC, CG.lookupSCC(A));
  EXPECT_EQ(&BC, CG.lookupSCC(B));
  EXPECT_EQ(&CC, CG.lookupSCC(C));
  EXPECT_EQ(&DC, CG.lookupSCC(D));
  EXPECT_EQ(&ARC, CG.lookupRefSCC(A));
  EXPECT_EQ(&BRC, CG.lookupRefSCC(B));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(C));
  EXPECT_EQ(&DRC, CG.lookupRefSCC(D));
}

TEST(LazyCallGraphTest, IncomingEdgeInsertion) {
  LLVMContext Context;
  // We want to ensure we can add edges even across complex diamond graphs, so
  // we use the diamond of triangles graph defined above. The ascii diagram is
  // repeated here for easy reference.
  //
  //         d1       |
  //        /  \      |
  //       d3--d2     |
  //      /     \     |
  //     b1     c1    |
  //   /  \    /  \   |
  //  b3--b2  c3--c2  |
  //       \  /       |
  //        a1        |
  //       /  \       |
  //      a3--a2      |
  //
  std::unique_ptr<Module> M = parseAssembly(Context, DiamondOfTriangles);
  LazyCallGraph CG = buildCG(*M);

  // Force the graph to be fully expanded.
  CG.buildRefSCCs();
  for (LazyCallGraph::RefSCC &RC : CG.postorder_ref_sccs())
    dbgs() << "Formed RefSCC: " << RC << "\n";

  LazyCallGraph::Node &A1 = *CG.lookup(lookupFunction(*M, "a1"));
  LazyCallGraph::Node &A2 = *CG.lookup(lookupFunction(*M, "a2"));
  LazyCallGraph::Node &A3 = *CG.lookup(lookupFunction(*M, "a3"));
  LazyCallGraph::Node &B1 = *CG.lookup(lookupFunction(*M, "b1"));
  LazyCallGraph::Node &B2 = *CG.lookup(lookupFunction(*M, "b2"));
  LazyCallGraph::Node &B3 = *CG.lookup(lookupFunction(*M, "b3"));
  LazyCallGraph::Node &C1 = *CG.lookup(lookupFunction(*M, "c1"));
  LazyCallGraph::Node &C2 = *CG.lookup(lookupFunction(*M, "c2"));
  LazyCallGraph::Node &C3 = *CG.lookup(lookupFunction(*M, "c3"));
  LazyCallGraph::Node &D1 = *CG.lookup(lookupFunction(*M, "d1"));
  LazyCallGraph::Node &D2 = *CG.lookup(lookupFunction(*M, "d2"));
  LazyCallGraph::Node &D3 = *CG.lookup(lookupFunction(*M, "d3"));
  LazyCallGraph::RefSCC &ARC = *CG.lookupRefSCC(A1);
  LazyCallGraph::RefSCC &BRC = *CG.lookupRefSCC(B1);
  LazyCallGraph::RefSCC &CRC = *CG.lookupRefSCC(C1);
  LazyCallGraph::RefSCC &DRC = *CG.lookupRefSCC(D1);
  ASSERT_EQ(&ARC, CG.lookupRefSCC(A2));
  ASSERT_EQ(&ARC, CG.lookupRefSCC(A3));
  ASSERT_EQ(&BRC, CG.lookupRefSCC(B2));
  ASSERT_EQ(&BRC, CG.lookupRefSCC(B3));
  ASSERT_EQ(&CRC, CG.lookupRefSCC(C2));
  ASSERT_EQ(&CRC, CG.lookupRefSCC(C3));
  ASSERT_EQ(&DRC, CG.lookupRefSCC(D2));
  ASSERT_EQ(&DRC, CG.lookupRefSCC(D3));
  ASSERT_EQ(1, std::distance(D2->begin(), D2->end()));

  // Add an edge to make the graph:
  //
  //         d1         |
  //        /  \        |
  //       d3--d2---.   |
  //      /     \    |  |
  //     b1     c1   |  |
  //   /  \    /  \ /   |
  //  b3--b2  c3--c2    |
  //       \  /         |
  //        a1          |
  //       /  \         |
  //      a3--a2        |
  auto MergedRCs = CRC.insertIncomingRefEdge(D2, C2);
  // Make sure we connected the nodes.
  for (LazyCallGraph::Edge E : *D2) {
    if (&E.getNode() == &D3)
      continue;
    EXPECT_EQ(&C2, &E.getNode());
  }
  // And marked the D ref-SCC as no longer valid.
  EXPECT_EQ(1u, MergedRCs.size());
  EXPECT_EQ(&DRC, MergedRCs[0]);

  // Make sure we have the correct nodes in the SCC sets.
  EXPECT_EQ(&ARC, CG.lookupRefSCC(A1));
  EXPECT_EQ(&ARC, CG.lookupRefSCC(A2));
  EXPECT_EQ(&ARC, CG.lookupRefSCC(A3));
  EXPECT_EQ(&BRC, CG.lookupRefSCC(B1));
  EXPECT_EQ(&BRC, CG.lookupRefSCC(B2));
  EXPECT_EQ(&BRC, CG.lookupRefSCC(B3));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(C1));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(C2));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(C3));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(D1));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(D2));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(D3));

  // And that ancestry tests have been updated.
  EXPECT_TRUE(ARC.isParentOf(CRC));
  EXPECT_TRUE(BRC.isParentOf(CRC));

  // And verify the post-order walk reflects the updated structure.
  auto I = CG.postorder_ref_scc_begin(), E = CG.postorder_ref_scc_end();
  ASSERT_NE(I, E);
  EXPECT_EQ(&CRC, &*I) << "Actual RefSCC: " << *I;
  ASSERT_NE(++I, E);
  EXPECT_EQ(&BRC, &*I) << "Actual RefSCC: " << *I;
  ASSERT_NE(++I, E);
  EXPECT_EQ(&ARC, &*I) << "Actual RefSCC: " << *I;
  EXPECT_EQ(++I, E);
}

TEST(LazyCallGraphTest, IncomingEdgeInsertionRefGraph) {
  LLVMContext Context;
  // Another variation of the above test but with all the edges switched to
  // references rather than calls.
  std::unique_ptr<Module> M =
      parseAssembly(Context, DiamondOfTrianglesRefGraph);
  LazyCallGraph CG = buildCG(*M);

  // Force the graph to be fully expanded.
  CG.buildRefSCCs();
  for (LazyCallGraph::RefSCC &RC : CG.postorder_ref_sccs())
    dbgs() << "Formed RefSCC: " << RC << "\n";

  LazyCallGraph::Node &A1 = *CG.lookup(lookupFunction(*M, "a1"));
  LazyCallGraph::Node &A2 = *CG.lookup(lookupFunction(*M, "a2"));
  LazyCallGraph::Node &A3 = *CG.lookup(lookupFunction(*M, "a3"));
  LazyCallGraph::Node &B1 = *CG.lookup(lookupFunction(*M, "b1"));
  LazyCallGraph::Node &B2 = *CG.lookup(lookupFunction(*M, "b2"));
  LazyCallGraph::Node &B3 = *CG.lookup(lookupFunction(*M, "b3"));
  LazyCallGraph::Node &C1 = *CG.lookup(lookupFunction(*M, "c1"));
  LazyCallGraph::Node &C2 = *CG.lookup(lookupFunction(*M, "c2"));
  LazyCallGraph::Node &C3 = *CG.lookup(lookupFunction(*M, "c3"));
  LazyCallGraph::Node &D1 = *CG.lookup(lookupFunction(*M, "d1"));
  LazyCallGraph::Node &D2 = *CG.lookup(lookupFunction(*M, "d2"));
  LazyCallGraph::Node &D3 = *CG.lookup(lookupFunction(*M, "d3"));
  LazyCallGraph::RefSCC &ARC = *CG.lookupRefSCC(A1);
  LazyCallGraph::RefSCC &BRC = *CG.lookupRefSCC(B1);
  LazyCallGraph::RefSCC &CRC = *CG.lookupRefSCC(C1);
  LazyCallGraph::RefSCC &DRC = *CG.lookupRefSCC(D1);
  ASSERT_EQ(&ARC, CG.lookupRefSCC(A2));
  ASSERT_EQ(&ARC, CG.lookupRefSCC(A3));
  ASSERT_EQ(&BRC, CG.lookupRefSCC(B2));
  ASSERT_EQ(&BRC, CG.lookupRefSCC(B3));
  ASSERT_EQ(&CRC, CG.lookupRefSCC(C2));
  ASSERT_EQ(&CRC, CG.lookupRefSCC(C3));
  ASSERT_EQ(&DRC, CG.lookupRefSCC(D2));
  ASSERT_EQ(&DRC, CG.lookupRefSCC(D3));
  ASSERT_EQ(1, std::distance(D2->begin(), D2->end()));

  // Add an edge to make the graph:
  //
  //         d1         |
  //        /  \        |
  //       d3--d2---.   |
  //      /     \    |  |
  //     b1     c1   |  |
  //   /  \    /  \ /   |
  //  b3--b2  c3--c2    |
  //       \  /         |
  //        a1          |
  //       /  \         |
  //      a3--a2        |
  auto MergedRCs = CRC.insertIncomingRefEdge(D2, C2);
  // Make sure we connected the nodes.
  for (LazyCallGraph::Edge E : *D2) {
    if (&E.getNode() == &D3)
      continue;
    EXPECT_EQ(&C2, &E.getNode());
  }
  // And marked the D ref-SCC as no longer valid.
  EXPECT_EQ(1u, MergedRCs.size());
  EXPECT_EQ(&DRC, MergedRCs[0]);

  // Make sure we have the correct nodes in the SCC sets.
  EXPECT_EQ(&ARC, CG.lookupRefSCC(A1));
  EXPECT_EQ(&ARC, CG.lookupRefSCC(A2));
  EXPECT_EQ(&ARC, CG.lookupRefSCC(A3));
  EXPECT_EQ(&BRC, CG.lookupRefSCC(B1));
  EXPECT_EQ(&BRC, CG.lookupRefSCC(B2));
  EXPECT_EQ(&BRC, CG.lookupRefSCC(B3));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(C1));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(C2));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(C3));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(D1));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(D2));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(D3));

  // And that ancestry tests have been updated.
  EXPECT_TRUE(ARC.isParentOf(CRC));
  EXPECT_TRUE(BRC.isParentOf(CRC));

  // And verify the post-order walk reflects the updated structure.
  auto I = CG.postorder_ref_scc_begin(), E = CG.postorder_ref_scc_end();
  ASSERT_NE(I, E);
  EXPECT_EQ(&CRC, &*I) << "Actual RefSCC: " << *I;
  ASSERT_NE(++I, E);
  EXPECT_EQ(&BRC, &*I) << "Actual RefSCC: " << *I;
  ASSERT_NE(++I, E);
  EXPECT_EQ(&ARC, &*I) << "Actual RefSCC: " << *I;
  EXPECT_EQ(++I, E);
}

TEST(LazyCallGraphTest, IncomingEdgeInsertionLargeCallCycle) {
  LLVMContext Context;
  std::unique_ptr<Module> M = parseAssembly(Context, "define void @a() {\n"
                                                     "entry:\n"
                                                     "  call void @b()\n"
                                                     "  ret void\n"
                                                     "}\n"
                                                     "define void @b() {\n"
                                                     "entry:\n"
                                                     "  call void @c()\n"
                                                     "  ret void\n"
                                                     "}\n"
                                                     "define void @c() {\n"
                                                     "entry:\n"
                                                     "  call void @d()\n"
                                                     "  ret void\n"
                                                     "}\n"
                                                     "define void @d() {\n"
                                                     "entry:\n"
                                                     "  ret void\n"
                                                     "}\n");
  LazyCallGraph CG = buildCG(*M);

  // Force the graph to be fully expanded.
  CG.buildRefSCCs();
  for (LazyCallGraph::RefSCC &RC : CG.postorder_ref_sccs())
    dbgs() << "Formed RefSCC: " << RC << "\n";

  LazyCallGraph::Node &A = *CG.lookup(lookupFunction(*M, "a"));
  LazyCallGraph::Node &B = *CG.lookup(lookupFunction(*M, "b"));
  LazyCallGraph::Node &C = *CG.lookup(lookupFunction(*M, "c"));
  LazyCallGraph::Node &D = *CG.lookup(lookupFunction(*M, "d"));
  LazyCallGraph::SCC &AC = *CG.lookupSCC(A);
  LazyCallGraph::SCC &BC = *CG.lookupSCC(B);
  LazyCallGraph::SCC &CC = *CG.lookupSCC(C);
  LazyCallGraph::SCC &DC = *CG.lookupSCC(D);
  LazyCallGraph::RefSCC &ARC = *CG.lookupRefSCC(A);
  LazyCallGraph::RefSCC &BRC = *CG.lookupRefSCC(B);
  LazyCallGraph::RefSCC &CRC = *CG.lookupRefSCC(C);
  LazyCallGraph::RefSCC &DRC = *CG.lookupRefSCC(D);

  // Connect the top to the bottom forming a large RefSCC made up mostly of calls.
  auto MergedRCs = ARC.insertIncomingRefEdge(D, A);
  // Make sure we connected the nodes.
  EXPECT_NE(D->begin(), D->end());
  EXPECT_EQ(&A, &D->begin()->getNode());

  // Check that we have the dead RCs, but ignore the order.
  EXPECT_EQ(3u, MergedRCs.size());
  EXPECT_NE(find(MergedRCs, &BRC), MergedRCs.end());
  EXPECT_NE(find(MergedRCs, &CRC), MergedRCs.end());
  EXPECT_NE(find(MergedRCs, &DRC), MergedRCs.end());

  // Make sure the nodes point to the right place now.
  EXPECT_EQ(&ARC, CG.lookupRefSCC(A));
  EXPECT_EQ(&ARC, CG.lookupRefSCC(B));
  EXPECT_EQ(&ARC, CG.lookupRefSCC(C));
  EXPECT_EQ(&ARC, CG.lookupRefSCC(D));

  // Check that the SCCs are in postorder.
  EXPECT_EQ(4, ARC.size());
  EXPECT_EQ(&DC, &ARC[0]);
  EXPECT_EQ(&CC, &ARC[1]);
  EXPECT_EQ(&BC, &ARC[2]);
  EXPECT_EQ(&AC, &ARC[3]);

  // And verify the post-order walk reflects the updated structure.
  auto I = CG.postorder_ref_scc_begin(), E = CG.postorder_ref_scc_end();
  ASSERT_NE(I, E);
  EXPECT_EQ(&ARC, &*I) << "Actual RefSCC: " << *I;
  EXPECT_EQ(++I, E);
}

TEST(LazyCallGraphTest, IncomingEdgeInsertionLargeRefCycle) {
  LLVMContext Context;
  std::unique_ptr<Module> M =
      parseAssembly(Context, "define void @a() {\n"
                             "entry:\n"
                             "  %p = alloca void ()*\n"
                             "  store void ()* @b, void ()** %p\n"
                             "  ret void\n"
                             "}\n"
                             "define void @b() {\n"
                             "entry:\n"
                             "  %p = alloca void ()*\n"
                             "  store void ()* @c, void ()** %p\n"
                             "  ret void\n"
                             "}\n"
                             "define void @c() {\n"
                             "entry:\n"
                             "  %p = alloca void ()*\n"
                             "  store void ()* @d, void ()** %p\n"
                             "  ret void\n"
                             "}\n"
                             "define void @d() {\n"
                             "entry:\n"
                             "  ret void\n"
                             "}\n");
  LazyCallGraph CG = buildCG(*M);

  // Force the graph to be fully expanded.
  CG.buildRefSCCs();
  for (LazyCallGraph::RefSCC &RC : CG.postorder_ref_sccs())
    dbgs() << "Formed RefSCC: " << RC << "\n";

  LazyCallGraph::Node &A = *CG.lookup(lookupFunction(*M, "a"));
  LazyCallGraph::Node &B = *CG.lookup(lookupFunction(*M, "b"));
  LazyCallGraph::Node &C = *CG.lookup(lookupFunction(*M, "c"));
  LazyCallGraph::Node &D = *CG.lookup(lookupFunction(*M, "d"));
  LazyCallGraph::RefSCC &ARC = *CG.lookupRefSCC(A);
  LazyCallGraph::RefSCC &BRC = *CG.lookupRefSCC(B);
  LazyCallGraph::RefSCC &CRC = *CG.lookupRefSCC(C);
  LazyCallGraph::RefSCC &DRC = *CG.lookupRefSCC(D);

  // Connect the top to the bottom forming a large RefSCC made up just of
  // references.
  auto MergedRCs = ARC.insertIncomingRefEdge(D, A);
  // Make sure we connected the nodes.
  EXPECT_NE(D->begin(), D->end());
  EXPECT_EQ(&A, &D->begin()->getNode());

  // Check that we have the dead RCs, but ignore the order.
  EXPECT_EQ(3u, MergedRCs.size());
  EXPECT_NE(find(MergedRCs, &BRC), MergedRCs.end());
  EXPECT_NE(find(MergedRCs, &CRC), MergedRCs.end());
  EXPECT_NE(find(MergedRCs, &DRC), MergedRCs.end());

  // Make sure the nodes point to the right place now.
  EXPECT_EQ(&ARC, CG.lookupRefSCC(A));
  EXPECT_EQ(&ARC, CG.lookupRefSCC(B));
  EXPECT_EQ(&ARC, CG.lookupRefSCC(C));
  EXPECT_EQ(&ARC, CG.lookupRefSCC(D));

  // And verify the post-order walk reflects the updated structure.
  auto I = CG.postorder_ref_scc_begin(), End = CG.postorder_ref_scc_end();
  ASSERT_NE(I, End);
  EXPECT_EQ(&ARC, &*I) << "Actual RefSCC: " << *I;
  EXPECT_EQ(++I, End);
}

TEST(LazyCallGraphTest, InlineAndDeleteFunction) {
  LLVMContext Context;
  // We want to ensure we can delete nodes from relatively complex graphs and
  // so use the diamond of triangles graph defined above.
  //
  // The ascii diagram is repeated here for easy reference.
  //
  //         d1       |
  //        /  \      |
  //       d3--d2     |
  //      /     \     |
  //     b1     c1    |
  //   /  \    /  \   |
  //  b3--b2  c3--c2  |
  //       \  /       |
  //        a1        |
  //       /  \       |
  //      a3--a2      |
  //
  std::unique_ptr<Module> M = parseAssembly(Context, DiamondOfTriangles);
  LazyCallGraph CG = buildCG(*M);

  // Force the graph to be fully expanded.
  CG.buildRefSCCs();
  for (LazyCallGraph::RefSCC &RC : CG.postorder_ref_sccs())
    dbgs() << "Formed RefSCC: " << RC << "\n";

  LazyCallGraph::Node &A1 = *CG.lookup(lookupFunction(*M, "a1"));
  LazyCallGraph::Node &A2 = *CG.lookup(lookupFunction(*M, "a2"));
  LazyCallGraph::Node &A3 = *CG.lookup(lookupFunction(*M, "a3"));
  LazyCallGraph::Node &B1 = *CG.lookup(lookupFunction(*M, "b1"));
  LazyCallGraph::Node &B2 = *CG.lookup(lookupFunction(*M, "b2"));
  LazyCallGraph::Node &B3 = *CG.lookup(lookupFunction(*M, "b3"));
  LazyCallGraph::Node &C1 = *CG.lookup(lookupFunction(*M, "c1"));
  LazyCallGraph::Node &C2 = *CG.lookup(lookupFunction(*M, "c2"));
  LazyCallGraph::Node &C3 = *CG.lookup(lookupFunction(*M, "c3"));
  LazyCallGraph::Node &D1 = *CG.lookup(lookupFunction(*M, "d1"));
  LazyCallGraph::Node &D2 = *CG.lookup(lookupFunction(*M, "d2"));
  LazyCallGraph::Node &D3 = *CG.lookup(lookupFunction(*M, "d3"));
  LazyCallGraph::RefSCC &ARC = *CG.lookupRefSCC(A1);
  LazyCallGraph::RefSCC &BRC = *CG.lookupRefSCC(B1);
  LazyCallGraph::RefSCC &CRC = *CG.lookupRefSCC(C1);
  LazyCallGraph::RefSCC &DRC = *CG.lookupRefSCC(D1);
  ASSERT_EQ(&ARC, CG.lookupRefSCC(A2));
  ASSERT_EQ(&ARC, CG.lookupRefSCC(A3));
  ASSERT_EQ(&BRC, CG.lookupRefSCC(B2));
  ASSERT_EQ(&BRC, CG.lookupRefSCC(B3));
  ASSERT_EQ(&CRC, CG.lookupRefSCC(C2));
  ASSERT_EQ(&CRC, CG.lookupRefSCC(C3));
  ASSERT_EQ(&DRC, CG.lookupRefSCC(D2));
  ASSERT_EQ(&DRC, CG.lookupRefSCC(D3));
  ASSERT_EQ(1, std::distance(D2->begin(), D2->end()));

  // Delete d2 from the graph, as if it had been inlined.
  //
  //         d1         |
  //        / /         |
  //       d3--.        |
  //      /     \       |
  //     b1     c1      |
  //   /  \    /  \     |
  //  b3--b2  c3--c2    |
  //       \  /         |
  //        a1          |
  //       /  \         |
  //      a3--a2        |

  Function &D2F = D2.getFunction();
  CallInst *C1Call = nullptr, *D1Call = nullptr;
  for (User *U : D2F.users()) {
    CallInst *CI = dyn_cast<CallInst>(U);
    ASSERT_TRUE(CI) << "Expected a call: " << *U;
    if (CI->getParent()->getParent() == &C1.getFunction()) {
      ASSERT_EQ(nullptr, C1Call) << "Found too many C1 calls: " << *CI;
      C1Call = CI;
    } else if (CI->getParent()->getParent() == &D1.getFunction()) {
      ASSERT_EQ(nullptr, D1Call) << "Found too many D1 calls: " << *CI;
      D1Call = CI;
    } else {
      FAIL() << "Found an unexpected call instruction: " << *CI;
    }
  }
  ASSERT_NE(C1Call, nullptr);
  ASSERT_NE(D1Call, nullptr);
  ASSERT_EQ(&D2F, C1Call->getCalledFunction());
  ASSERT_EQ(&D2F, D1Call->getCalledFunction());
  C1Call->setCalledFunction(&D3.getFunction());
  D1Call->setCalledFunction(&D3.getFunction());
  ASSERT_EQ(0u, D2F.getNumUses());

  // Insert new edges first.
  CRC.insertTrivialCallEdge(C1, D3);
  DRC.insertTrivialCallEdge(D1, D3);

  // Then remove the old ones.
  LazyCallGraph::SCC &DC = *CG.lookupSCC(D2);
  auto NewCs = DRC.switchInternalEdgeToRef(D1, D2);
  EXPECT_EQ(&DC, CG.lookupSCC(D2));
  EXPECT_EQ(NewCs.end(), std::next(NewCs.begin()));
  LazyCallGraph::SCC &NewDC = *NewCs.begin();
  EXPECT_EQ(&NewDC, CG.lookupSCC(D1));
  EXPECT_EQ(&NewDC, CG.lookupSCC(D3));
  auto NewRCs = DRC.removeInternalRefEdge(D1, {&D2});
  ASSERT_EQ(2u, NewRCs.size());
  LazyCallGraph::RefSCC &NewDRC = *NewRCs[0];
  EXPECT_EQ(&NewDRC, CG.lookupRefSCC(D1));
  EXPECT_EQ(&NewDRC, CG.lookupRefSCC(D3));
  LazyCallGraph::RefSCC &D2RC = *NewRCs[1];
  EXPECT_EQ(&D2RC, CG.lookupRefSCC(D2));
  EXPECT_FALSE(NewDRC.isParentOf(D2RC));
  EXPECT_TRUE(CRC.isParentOf(D2RC));
  EXPECT_TRUE(CRC.isParentOf(NewDRC));
  EXPECT_TRUE(D2RC.isParentOf(NewDRC));
  CRC.removeOutgoingEdge(C1, D2);
  EXPECT_FALSE(CRC.isParentOf(D2RC));
  EXPECT_TRUE(CRC.isParentOf(NewDRC));
  EXPECT_TRUE(D2RC.isParentOf(NewDRC));

  // Now that we've updated the call graph, D2 is dead, so remove it.
  CG.removeDeadFunction(D2F);

  // Check that the graph still looks the same.
  EXPECT_EQ(&ARC, CG.lookupRefSCC(A1));
  EXPECT_EQ(&ARC, CG.lookupRefSCC(A2));
  EXPECT_EQ(&ARC, CG.lookupRefSCC(A3));
  EXPECT_EQ(&BRC, CG.lookupRefSCC(B1));
  EXPECT_EQ(&BRC, CG.lookupRefSCC(B2));
  EXPECT_EQ(&BRC, CG.lookupRefSCC(B3));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(C1));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(C2));
  EXPECT_EQ(&CRC, CG.lookupRefSCC(C3));
  EXPECT_EQ(&NewDRC, CG.lookupRefSCC(D1));
  EXPECT_EQ(&NewDRC, CG.lookupRefSCC(D3));
  EXPECT_TRUE(CRC.isParentOf(NewDRC));

  // Verify the post-order walk hasn't changed.
  auto I = CG.postorder_ref_scc_begin(), E = CG.postorder_ref_scc_end();
  ASSERT_NE(I, E);
  EXPECT_EQ(&NewDRC, &*I) << "Actual RefSCC: " << *I;
  ASSERT_NE(++I, E);
  EXPECT_EQ(&CRC, &*I) << "Actual RefSCC: " << *I;
  ASSERT_NE(++I, E);
  EXPECT_EQ(&BRC, &*I) << "Actual RefSCC: " << *I;
  ASSERT_NE(++I, E);
  EXPECT_EQ(&ARC, &*I) << "Actual RefSCC: " << *I;
  EXPECT_EQ(++I, E);
}

TEST(LazyCallGraphTest, InternalEdgeMutation) {
  LLVMContext Context;
  std::unique_ptr<Module> M = parseAssembly(Context, "define void @a() {\n"
                                                     "entry:\n"
                                                     "  call void @b()\n"
                                                     "  ret void\n"
                                                     "}\n"
                                                     "define void @b() {\n"
                                                     "entry:\n"
                                                     "  call void @c()\n"
                                                     "  ret void\n"
                                                     "}\n"
                                                     "define void @c() {\n"
                                                     "entry:\n"
                                                     "  call void @a()\n"
                                                     "  ret void\n"
                                                     "}\n");
  LazyCallGraph CG = buildCG(*M);

  // Force the graph to be fully expanded.
  CG.buildRefSCCs();
  auto I = CG.postorder_ref_scc_begin();
  LazyCallGraph::RefSCC &RC = *I++;
  EXPECT_EQ(CG.postorder_ref_scc_end(), I);

  LazyCallGraph::Node &A = *CG.lookup(lookupFunction(*M, "a"));
  LazyCallGraph::Node &B = *CG.lookup(lookupFunction(*M, "b"));
  LazyCallGraph::Node &C = *CG.lookup(lookupFunction(*M, "c"));
  EXPECT_EQ(&RC, CG.lookupRefSCC(A));
  EXPECT_EQ(&RC, CG.lookupRefSCC(B));
  EXPECT_EQ(&RC, CG.lookupRefSCC(C));
  EXPECT_EQ(1, RC.size());
  EXPECT_EQ(&*RC.begin(), CG.lookupSCC(A));
  EXPECT_EQ(&*RC.begin(), CG.lookupSCC(B));
  EXPECT_EQ(&*RC.begin(), CG.lookupSCC(C));

  // Insert an edge from 'a' to 'c'. Nothing changes about the graph.
  RC.insertInternalRefEdge(A, C);
  EXPECT_EQ(2, std::distance(A->begin(), A->end()));
  EXPECT_EQ(&RC, CG.lookupRefSCC(A));
  EXPECT_EQ(&RC, CG.lookupRefSCC(B));
  EXPECT_EQ(&RC, CG.lookupRefSCC(C));
  EXPECT_EQ(1, RC.size());
  EXPECT_EQ(&*RC.begin(), CG.lookupSCC(A));
  EXPECT_EQ(&*RC.begin(), CG.lookupSCC(B));
  EXPECT_EQ(&*RC.begin(), CG.lookupSCC(C));

  // Switch the call edge from 'b' to 'c' to a ref edge. This will break the
  // call cycle and cause us to form more SCCs. The RefSCC will remain the same
  // though.
  auto NewCs = RC.switchInternalEdgeToRef(B, C);
  EXPECT_EQ(&RC, CG.lookupRefSCC(A));
  EXPECT_EQ(&RC, CG.lookupRefSCC(B));
  EXPECT_EQ(&RC, CG.lookupRefSCC(C));
  auto J = RC.begin();
  // The SCCs must be in *post-order* which means successors before
  // predecessors. At this point we have call edges from C to A and from A to
  // B. The only valid postorder is B, A, C.
  EXPECT_EQ(&*J++, CG.lookupSCC(B));
  EXPECT_EQ(&*J++, CG.lookupSCC(A));
  EXPECT_EQ(&*J++, CG.lookupSCC(C));
  EXPECT_EQ(RC.end(), J);
  // And the returned range must be the slice of this sequence containing new
  // SCCs.
  EXPECT_EQ(RC.begin(), NewCs.begin());
  EXPECT_EQ(std::prev(RC.end()), NewCs.end());

  // Test turning the ref edge from A to C into a call edge. This will form an
  // SCC out of A and C. Since we previously had a call edge from C to A, the
  // C SCC should be preserved and have A merged into it while the A SCC should
  // be invalidated.
  LazyCallGraph::SCC &AC = *CG.lookupSCC(A);
  LazyCallGraph::SCC &CC = *CG.lookupSCC(C);
  EXPECT_TRUE(RC.switchInternalEdgeToCall(A, C, [&](ArrayRef<LazyCallGraph::SCC *> MergedCs) {
    ASSERT_EQ(1u, MergedCs.size());
    EXPECT_EQ(&AC, MergedCs[0]);
  }));
  EXPECT_EQ(2, CC.size());
  EXPECT_EQ(&CC, CG.lookupSCC(A));
  EXPECT_EQ(&CC, CG.lookupSCC(C));
  J = RC.begin();
  EXPECT_EQ(&*J++, CG.lookupSCC(B));
  EXPECT_EQ(&*J++, CG.lookupSCC(C));
  EXPECT_EQ(RC.end(), J);
}

TEST(LazyCallGraphTest, InternalEdgeRemoval) {
  LLVMContext Context;
  // A nice fully connected (including self-edges) RefSCC.
  std::unique_ptr<Module> M = parseAssembly(
      Context, "define void @a(i8** %ptr) {\n"
               "entry:\n"
               "  store i8* bitcast (void(i8**)* @a to i8*), i8** %ptr\n"
               "  store i8* bitcast (void(i8**)* @b to i8*), i8** %ptr\n"
               "  store i8* bitcast (void(i8**)* @c to i8*), i8** %ptr\n"
               "  ret void\n"
               "}\n"
               "define void @b(i8** %ptr) {\n"
               "entry:\n"
               "  store i8* bitcast (void(i8**)* @a to i8*), i8** %ptr\n"
               "  store i8* bitcast (void(i8**)* @b to i8*), i8** %ptr\n"
               "  store i8* bitcast (void(i8**)* @c to i8*), i8** %ptr\n"
               "  ret void\n"
               "}\n"
               "define void @c(i8** %ptr) {\n"
               "entry:\n"
               "  store i8* bitcast (void(i8**)* @a to i8*), i8** %ptr\n"
               "  store i8* bitcast (void(i8**)* @b to i8*), i8** %ptr\n"
               "  store i8* bitcast (void(i8**)* @c to i8*), i8** %ptr\n"
               "  ret void\n"
               "}\n");
  LazyCallGraph CG = buildCG(*M);

  // Force the graph to be fully expanded.
  CG.buildRefSCCs();
  auto I = CG.postorder_ref_scc_begin(), E = CG.postorder_ref_scc_end();
  LazyCallGraph::RefSCC &RC = *I;
  EXPECT_EQ(E, std::next(I));

  LazyCallGraph::Node &A = *CG.lookup(lookupFunction(*M, "a"));
  LazyCallGraph::Node &B = *CG.lookup(lookupFunction(*M, "b"));
  LazyCallGraph::Node &C = *CG.lookup(lookupFunction(*M, "c"));
  EXPECT_EQ(&RC, CG.lookupRefSCC(A));
  EXPECT_EQ(&RC, CG.lookupRefSCC(B));
  EXPECT_EQ(&RC, CG.lookupRefSCC(C));

  // Remove the edge from b -> a, which should leave the 3 functions still in
  // a single connected component because of a -> b -> c -> a.
  SmallVector<LazyCallGraph::RefSCC *, 1> NewRCs =
      RC.removeInternalRefEdge(B, {&A});
  EXPECT_EQ(0u, NewRCs.size());
  EXPECT_EQ(&RC, CG.lookupRefSCC(A));
  EXPECT_EQ(&RC, CG.lookupRefSCC(B));
  EXPECT_EQ(&RC, CG.lookupRefSCC(C));
  auto J = CG.postorder_ref_scc_begin();
  EXPECT_EQ(I, J);
  EXPECT_EQ(&RC, &*J);
  EXPECT_EQ(E, std::next(J));

  // Increment I before we actually mutate the structure so that it remains
  // a valid iterator.
  ++I;

  // Remove the edge from c -> a, which should leave 'a' in the original RefSCC
  // and form a new RefSCC for 'b' and 'c'.
  NewRCs = RC.removeInternalRefEdge(C, {&A});
  ASSERT_EQ(2u, NewRCs.size());
  LazyCallGraph::RefSCC &BCRC = *NewRCs[0];
  LazyCallGraph::RefSCC &ARC = *NewRCs[1];
  EXPECT_EQ(&ARC, CG.lookupRefSCC(A));
  EXPECT_EQ(1, std::distance(ARC.begin(), ARC.end()));
  EXPECT_EQ(&BCRC, CG.lookupRefSCC(B));
  EXPECT_EQ(&BCRC, CG.lookupRefSCC(C));
  J = CG.postorder_ref_scc_begin();
  EXPECT_NE(I, J);
  EXPECT_EQ(&BCRC, &*J);
  ++J;
  EXPECT_NE(I, J);
  EXPECT_EQ(&ARC, &*J);
  ++J;
  EXPECT_EQ(I, J);
  EXPECT_EQ(E, J);
}

TEST(LazyCallGraphTest, InternalMultiEdgeRemoval) {
  LLVMContext Context;
  // A nice fully connected (including self-edges) RefSCC.
  std::unique_ptr<Module> M = parseAssembly(
      Context, "define void @a(i8** %ptr) {\n"
               "entry:\n"
               "  store i8* bitcast (void(i8**)* @a to i8*), i8** %ptr\n"
               "  store i8* bitcast (void(i8**)* @b to i8*), i8** %ptr\n"
               "  store i8* bitcast (void(i8**)* @c to i8*), i8** %ptr\n"
               "  ret void\n"
               "}\n"
               "define void @b(i8** %ptr) {\n"
               "entry:\n"
               "  store i8* bitcast (void(i8**)* @a to i8*), i8** %ptr\n"
               "  store i8* bitcast (void(i8**)* @b to i8*), i8** %ptr\n"
               "  store i8* bitcast (void(i8**)* @c to i8*), i8** %ptr\n"
               "  ret void\n"
               "}\n"
               "define void @c(i8** %ptr) {\n"
               "entry:\n"
               "  store i8* bitcast (void(i8**)* @a to i8*), i8** %ptr\n"
               "  store i8* bitcast (void(i8**)* @b to i8*), i8** %ptr\n"
               "  store i8* bitcast (void(i8**)* @c to i8*), i8** %ptr\n"
               "  ret void\n"
               "}\n");
  LazyCallGraph CG = buildCG(*M);

  // Force the graph to be fully expanded.
  CG.buildRefSCCs();
  auto I = CG.postorder_ref_scc_begin(), E = CG.postorder_ref_scc_end();
  LazyCallGraph::RefSCC &RC = *I;
  EXPECT_EQ(E, std::next(I));

  LazyCallGraph::Node &A = *CG.lookup(lookupFunction(*M, "a"));
  LazyCallGraph::Node &B = *CG.lookup(lookupFunction(*M, "b"));
  LazyCallGraph::Node &C = *CG.lookup(lookupFunction(*M, "c"));
  EXPECT_EQ(&RC, CG.lookupRefSCC(A));
  EXPECT_EQ(&RC, CG.lookupRefSCC(B));
  EXPECT_EQ(&RC, CG.lookupRefSCC(C));

  // Increment I before we actually mutate the structure so that it remains
  // a valid iterator.
  ++I;

  // Remove the edges from b -> a and b -> c, leaving b in its own RefSCC.
  SmallVector<LazyCallGraph::RefSCC *, 1> NewRCs =
      RC.removeInternalRefEdge(B, {&A, &C});

  ASSERT_EQ(2u, NewRCs.size());
  LazyCallGraph::RefSCC &BRC = *NewRCs[0];
  LazyCallGraph::RefSCC &ACRC = *NewRCs[1];
  EXPECT_EQ(&BRC, CG.lookupRefSCC(B));
  EXPECT_EQ(1, std::distance(BRC.begin(), BRC.end()));
  EXPECT_EQ(&ACRC, CG.lookupRefSCC(A));
  EXPECT_EQ(&ACRC, CG.lookupRefSCC(C));
  auto J = CG.postorder_ref_scc_begin();
  EXPECT_NE(I, J);
  EXPECT_EQ(&BRC, &*J);
  ++J;
  EXPECT_NE(I, J);
  EXPECT_EQ(&ACRC, &*J);
  ++J;
  EXPECT_EQ(I, J);
  EXPECT_EQ(E, J);
}

TEST(LazyCallGraphTest, InternalNoOpEdgeRemoval) {
  LLVMContext Context;
  // A graph with a single cycle formed both from call and reference edges
  // which makes the reference edges trivial to delete. The graph looks like:
  //
  // Reference edges: a -> b -> c -> a
  //      Call edges: a -> c -> b -> a
  std::unique_ptr<Module> M = parseAssembly(
      Context, "define void @a(i8** %ptr) {\n"
               "entry:\n"
               "  call void @b(i8** %ptr)\n"
               "  store i8* bitcast (void(i8**)* @c to i8*), i8** %ptr\n"
               "  ret void\n"
               "}\n"
               "define void @b(i8** %ptr) {\n"
               "entry:\n"
               "  store i8* bitcast (void(i8**)* @a to i8*), i8** %ptr\n"
               "  call void @c(i8** %ptr)\n"
               "  ret void\n"
               "}\n"
               "define void @c(i8** %ptr) {\n"
               "entry:\n"
               "  call void @a(i8** %ptr)\n"
               "  store i8* bitcast (void(i8**)* @b to i8*), i8** %ptr\n"
               "  ret void\n"
               "}\n");
  LazyCallGraph CG = buildCG(*M);

  // Force the graph to be fully expanded.
  CG.buildRefSCCs();
  auto I = CG.postorder_ref_scc_begin(), E = CG.postorder_ref_scc_end();
  LazyCallGraph::RefSCC &RC = *I;
  EXPECT_EQ(E, std::next(I));

  LazyCallGraph::SCC &C = *RC.begin();
  EXPECT_EQ(RC.end(), std::next(RC.begin()));

  LazyCallGraph::Node &AN = *CG.lookup(lookupFunction(*M, "a"));
  LazyCallGraph::Node &BN = *CG.lookup(lookupFunction(*M, "b"));
  LazyCallGraph::Node &CN = *CG.lookup(lookupFunction(*M, "c"));
  EXPECT_EQ(&RC, CG.lookupRefSCC(AN));
  EXPECT_EQ(&RC, CG.lookupRefSCC(BN));
  EXPECT_EQ(&RC, CG.lookupRefSCC(CN));
  EXPECT_EQ(&C, CG.lookupSCC(AN));
  EXPECT_EQ(&C, CG.lookupSCC(BN));
  EXPECT_EQ(&C, CG.lookupSCC(CN));

  // Remove the edge from a -> c which doesn't change anything.
  SmallVector<LazyCallGraph::RefSCC *, 1> NewRCs =
      RC.removeInternalRefEdge(AN, {&CN});
  EXPECT_EQ(0u, NewRCs.size());
  EXPECT_EQ(&RC, CG.lookupRefSCC(AN));
  EXPECT_EQ(&RC, CG.lookupRefSCC(BN));
  EXPECT_EQ(&RC, CG.lookupRefSCC(CN));
  EXPECT_EQ(&C, CG.lookupSCC(AN));
  EXPECT_EQ(&C, CG.lookupSCC(BN));
  EXPECT_EQ(&C, CG.lookupSCC(CN));
  auto J = CG.postorder_ref_scc_begin();
  EXPECT_EQ(I, J);
  EXPECT_EQ(&RC, &*J);
  EXPECT_EQ(E, std::next(J));

  // Remove the edge from b -> a and c -> b; again this doesn't change
  // anything.
  NewRCs = RC.removeInternalRefEdge(BN, {&AN});
  NewRCs = RC.removeInternalRefEdge(CN, {&BN});
  EXPECT_EQ(0u, NewRCs.size());
  EXPECT_EQ(&RC, CG.lookupRefSCC(AN));
  EXPECT_EQ(&RC, CG.lookupRefSCC(BN));
  EXPECT_EQ(&RC, CG.lookupRefSCC(CN));
  EXPECT_EQ(&C, CG.lookupSCC(AN));
  EXPECT_EQ(&C, CG.lookupSCC(BN));
  EXPECT_EQ(&C, CG.lookupSCC(CN));
  J = CG.postorder_ref_scc_begin();
  EXPECT_EQ(I, J);
  EXPECT_EQ(&RC, &*J);
  EXPECT_EQ(E, std::next(J));
}

TEST(LazyCallGraphTest, InternalCallEdgeToRef) {
  LLVMContext Context;
  // A nice fully connected (including self-edges) SCC (and RefSCC)
  std::unique_ptr<Module> M = parseAssembly(Context, "define void @a() {\n"
                                                     "entry:\n"
                                                     "  call void @a()\n"
                                                     "  call void @b()\n"
                                                     "  call void @c()\n"
                                                     "  ret void\n"
                                                     "}\n"
                                                     "define void @b() {\n"
                                                     "entry:\n"
                                                     "  call void @a()\n"
                                                     "  call void @b()\n"
                                                     "  call void @c()\n"
                                                     "  ret void\n"
                                                     "}\n"
                                                     "define void @c() {\n"
                                                     "entry:\n"
                                                     "  call void @a()\n"
                                                     "  call void @b()\n"
                                                     "  call void @c()\n"
                                                     "  ret void\n"
                                                     "}\n");
  LazyCallGraph CG = buildCG(*M);

  // Force the graph to be fully expanded.
  CG.buildRefSCCs();
  auto I = CG.postorder_ref_scc_begin();
  LazyCallGraph::RefSCC &RC = *I++;
  EXPECT_EQ(CG.postorder_ref_scc_end(), I);

  EXPECT_EQ(1, RC.size());
  LazyCallGraph::SCC &AC = *RC.begin();

  LazyCallGraph::Node &AN = *CG.lookup(lookupFunction(*M, "a"));
  LazyCallGraph::Node &BN = *CG.lookup(lookupFunction(*M, "b"));
  LazyCallGraph::Node &CN = *CG.lookup(lookupFunction(*M, "c"));
  EXPECT_EQ(&AC, CG.lookupSCC(AN));
  EXPECT_EQ(&AC, CG.lookupSCC(BN));
  EXPECT_EQ(&AC, CG.lookupSCC(CN));

  // Remove the call edge from b -> a to a ref edge, which should leave the
  // 3 functions still in a single connected component because of a -> b ->
  // c -> a.
  auto NewCs = RC.switchInternalEdgeToRef(BN, AN);
  EXPECT_EQ(NewCs.begin(), NewCs.end());
  EXPECT_EQ(1, RC.size());
  EXPECT_EQ(&AC, CG.lookupSCC(AN));
  EXPECT_EQ(&AC, CG.lookupSCC(BN));
  EXPECT_EQ(&AC, CG.lookupSCC(CN));

  // Remove the edge from c -> a, which should leave 'a' in the original SCC
  // and form a new SCC for 'b' and 'c'.
  NewCs = RC.switchInternalEdgeToRef(CN, AN);
  EXPECT_EQ(1, std::distance(NewCs.begin(), NewCs.end()));
  EXPECT_EQ(2, RC.size());
  EXPECT_EQ(&AC, CG.lookupSCC(AN));
  LazyCallGraph::SCC &BC = *CG.lookupSCC(BN);
  EXPECT_NE(&BC, &AC);
  EXPECT_EQ(&BC, CG.lookupSCC(CN));
  auto J = RC.find(AC);
  EXPECT_EQ(&AC, &*J);
  --J;
  EXPECT_EQ(&BC, &*J);
  EXPECT_EQ(RC.begin(), J);
  EXPECT_EQ(J, NewCs.begin());

  // Remove the edge from c -> b, which should leave 'b' in the original SCC
  // and form a new SCC for 'c'. It shouldn't change 'a's SCC.
  NewCs = RC.switchInternalEdgeToRef(CN, BN);
  EXPECT_EQ(1, std::distance(NewCs.begin(), NewCs.end()));
  EXPECT_EQ(3, RC.size());
  EXPECT_EQ(&AC, CG.lookupSCC(AN));
  EXPECT_EQ(&BC, CG.lookupSCC(BN));
  LazyCallGraph::SCC &CC = *CG.lookupSCC(CN);
  EXPECT_NE(&CC, &AC);
  EXPECT_NE(&CC, &BC);
  J = RC.find(AC);
  EXPECT_EQ(&AC, &*J);
  --J;
  EXPECT_EQ(&BC, &*J);
  --J;
  EXPECT_EQ(&CC, &*J);
  EXPECT_EQ(RC.begin(), J);
  EXPECT_EQ(J, NewCs.begin());
}

TEST(LazyCallGraphTest, InternalRefEdgeToCall) {
  LLVMContext Context;
  // Basic tests for making a ref edge a call. This hits the basics of the
  // process only.
  std::unique_ptr<Module> M =
      parseAssembly(Context, "define void @a() {\n"
                             "entry:\n"
                             "  call void @b()\n"
                             "  call void @c()\n"
                             "  store void()* @d, void()** undef\n"
                             "  ret void\n"
                             "}\n"
                             "define void @b() {\n"
                             "entry:\n"
                             "  store void()* @c, void()** undef\n"
                             "  call void @d()\n"
                             "  ret void\n"
                             "}\n"
                             "define void @c() {\n"
                             "entry:\n"
                             "  store void()* @b, void()** undef\n"
                             "  call void @d()\n"
                             "  ret void\n"
                             "}\n"
                             "define void @d() {\n"
                             "entry:\n"
                             "  store void()* @a, void()** undef\n"
                             "  ret void\n"
                             "}\n");
  LazyCallGraph CG = buildCG(*M);

  // Force the graph to be fully expanded.
  CG.buildRefSCCs();
  auto I = CG.postorder_ref_scc_begin();
  LazyCallGraph::RefSCC &RC = *I++;
  EXPECT_EQ(CG.postorder_ref_scc_end(), I);

  LazyCallGraph::Node &A = *CG.lookup(lookupFunction(*M, "a"));
  LazyCallGraph::Node &B = *CG.lookup(lookupFunction(*M, "b"));
  LazyCallGraph::Node &C = *CG.lookup(lookupFunction(*M, "c"));
  LazyCallGraph::Node &D = *CG.lookup(lookupFunction(*M, "d"));
  LazyCallGraph::SCC &AC = *CG.lookupSCC(A);
  LazyCallGraph::SCC &BC = *CG.lookupSCC(B);
  LazyCallGraph::SCC &CC = *CG.lookupSCC(C);
  LazyCallGraph::SCC &DC = *CG.lookupSCC(D);

  // Check the initial post-order. Note that B and C could be flipped here (and
  // in our mutation) without changing the nature of this test.
  ASSERT_EQ(4, RC.size());
  EXPECT_EQ(&DC, &RC[0]);
  EXPECT_EQ(&BC, &RC[1]);
  EXPECT_EQ(&CC, &RC[2]);
  EXPECT_EQ(&AC, &RC[3]);

  // Switch the ref edge from A -> D to a call edge. This should have no
  // effect as it is already in postorder and no new cycles are formed.
  EXPECT_FALSE(RC.switchInternalEdgeToCall(A, D));
  ASSERT_EQ(4, RC.size());
  EXPECT_EQ(&DC, &RC[0]);
  EXPECT_EQ(&BC, &RC[1]);
  EXPECT_EQ(&CC, &RC[2]);
  EXPECT_EQ(&AC, &RC[3]);

  // Switch B -> C to a call edge. This doesn't form any new cycles but does
  // require reordering the SCCs.
  EXPECT_FALSE(RC.switchInternalEdgeToCall(B, C));
  ASSERT_EQ(4, RC.size());
  EXPECT_EQ(&DC, &RC[0]);
  EXPECT_EQ(&CC, &RC[1]);
  EXPECT_EQ(&BC, &RC[2]);
  EXPECT_EQ(&AC, &RC[3]);

  // Switch C -> B to a call edge. This forms a cycle and forces merging SCCs.
  EXPECT_TRUE(RC.switchInternalEdgeToCall(C, B, [&](ArrayRef<LazyCallGraph::SCC *> MergedCs) {
    ASSERT_EQ(1u, MergedCs.size());
    EXPECT_EQ(&CC, MergedCs[0]);
  }));
  ASSERT_EQ(3, RC.size());
  EXPECT_EQ(&DC, &RC[0]);
  EXPECT_EQ(&BC, &RC[1]);
  EXPECT_EQ(&AC, &RC[2]);
  EXPECT_EQ(2, BC.size());
  EXPECT_EQ(&BC, CG.lookupSCC(B));
  EXPECT_EQ(&BC, CG.lookupSCC(C));
}

TEST(LazyCallGraphTest, InternalRefEdgeToCallNoCycleInterleaved) {
  LLVMContext Context;
  // Test for having a post-order prior to changing a ref edge to a call edge
  // with SCCs connecting to the source and connecting to the target, but not
  // connecting to both, interleaved between the source and target. This
  // ensures we correctly partition the range rather than simply moving one or
  // the other.
  std::unique_ptr<Module> M =
      parseAssembly(Context, "define void @a() {\n"
                             "entry:\n"
                             "  call void @b1()\n"
                             "  call void @c1()\n"
                             "  ret void\n"
                             "}\n"
                             "define void @b1() {\n"
                             "entry:\n"
                             "  call void @c1()\n"
                             "  call void @b2()\n"
                             "  ret void\n"
                             "}\n"
                             "define void @c1() {\n"
                             "entry:\n"
                             "  call void @b2()\n"
                             "  call void @c2()\n"
                             "  ret void\n"
                             "}\n"
                             "define void @b2() {\n"
                             "entry:\n"
                             "  call void @c2()\n"
                             "  call void @b3()\n"
                             "  ret void\n"
                             "}\n"
                             "define void @c2() {\n"
                             "entry:\n"
                             "  call void @b3()\n"
                             "  call void @c3()\n"
                             "  ret void\n"
                             "}\n"
                             "define void @b3() {\n"
                             "entry:\n"
                             "  call void @c3()\n"
                             "  call void @d()\n"
                             "  ret void\n"
                             "}\n"
                             "define void @c3() {\n"
                             "entry:\n"
                             "  store void()* @b1, void()** undef\n"
                             "  call void @d()\n"
                             "  ret void\n"
                             "}\n"
                             "define void @d() {\n"
                             "entry:\n"
                             "  store void()* @a, void()** undef\n"
                             "  ret void\n"
                             "}\n");
  LazyCallGraph CG = buildCG(*M);

  // Force the graph to be fully expanded.
  CG.buildRefSCCs();
  auto I = CG.postorder_ref_scc_begin();
  LazyCallGraph::RefSCC &RC = *I++;
  EXPECT_EQ(CG.postorder_ref_scc_end(), I);

  LazyCallGraph::Node &A = *CG.lookup(lookupFunction(*M, "a"));
  LazyCallGraph::Node &B1 = *CG.lookup(lookupFunction(*M, "b1"));
  LazyCallGraph::Node &B2 = *CG.lookup(lookupFunction(*M, "b2"));
  LazyCallGraph::Node &B3 = *CG.lookup(lookupFunction(*M, "b3"));
  LazyCallGraph::Node &C1 = *CG.lookup(lookupFunction(*M, "c1"));
  LazyCallGraph::Node &C2 = *CG.lookup(lookupFunction(*M, "c2"));
  LazyCallGraph::Node &C3 = *CG.lookup(lookupFunction(*M, "c3"));
  LazyCallGraph::Node &D = *CG.lookup(lookupFunction(*M, "d"));
  LazyCallGraph::SCC &AC = *CG.lookupSCC(A);
  LazyCallGraph::SCC &B1C = *CG.lookupSCC(B1);
  LazyCallGraph::SCC &B2C = *CG.lookupSCC(B2);
  LazyCallGraph::SCC &B3C = *CG.lookupSCC(B3);
  LazyCallGraph::SCC &C1C = *CG.lookupSCC(C1);
  LazyCallGraph::SCC &C2C = *CG.lookupSCC(C2);
  LazyCallGraph::SCC &C3C = *CG.lookupSCC(C3);
  LazyCallGraph::SCC &DC = *CG.lookupSCC(D);

  // Several call edges are initially present to force a particual post-order.
  // Remove them now, leaving an interleaved post-order pattern.
  RC.switchTrivialInternalEdgeToRef(B3, C3);
  RC.switchTrivialInternalEdgeToRef(C2, B3);
  RC.switchTrivialInternalEdgeToRef(B2, C2);
  RC.switchTrivialInternalEdgeToRef(C1, B2);
  RC.switchTrivialInternalEdgeToRef(B1, C1);

  // Check the initial post-order. We ensure this order with the extra edges
  // that are nuked above.
  ASSERT_EQ(8, RC.size());
  EXPECT_EQ(&DC, &RC[0]);
  EXPECT_EQ(&C3C, &RC[1]);
  EXPECT_EQ(&B3C, &RC[2]);
  EXPECT_EQ(&C2C, &RC[3]);
  EXPECT_EQ(&B2C, &RC[4]);
  EXPECT_EQ(&C1C, &RC[5]);
  EXPECT_EQ(&B1C, &RC[6]);
  EXPECT_EQ(&AC, &RC[7]);

  // Switch C3 -> B1 to a call edge. This doesn't form any new cycles but does
  // require reordering the SCCs in the face of tricky internal node
  // structures.
  EXPECT_FALSE(RC.switchInternalEdgeToCall(C3, B1));
  ASSERT_EQ(8, RC.size());
  EXPECT_EQ(&DC, &RC[0]);
  EXPECT_EQ(&B3C, &RC[1]);
  EXPECT_EQ(&B2C, &RC[2]);
  EXPECT_EQ(&B1C, &RC[3]);
  EXPECT_EQ(&C3C, &RC[4]);
  EXPECT_EQ(&C2C, &RC[5]);
  EXPECT_EQ(&C1C, &RC[6]);
  EXPECT_EQ(&AC, &RC[7]);
}

TEST(LazyCallGraphTest, InternalRefEdgeToCallBothPartitionAndMerge) {
  LLVMContext Context;
  // Test for having a postorder where between the source and target are all
  // three kinds of other SCCs:
  // 1) One connected to the target only that have to be shifted below the
  //    source.
  // 2) One connected to the source only that have to be shifted below the
  //    target.
  // 3) One connected to both source and target that has to remain and get
  //    merged away.
  //
  // To achieve this we construct a heavily connected graph to force
  // a particular post-order. Then we remove the forcing edges and connect
  // a cycle.
  //
  // Diagram for the graph we want on the left and the graph we use to force
  // the ordering on the right. Edges ponit down or right.
  //
  //   A    |    A    |
  //  / \   |   / \   |
  // B   E  |  B   \  |
  // |\  |  |  |\  |  |
  // | D |  |  C-D-E  |
  // |  \|  |  |  \|  |
  // C   F  |  \   F  |
  //  \ /   |   \ /   |
  //   G    |    G    |
  //
  // And we form a cycle by connecting F to B.
  std::unique_ptr<Module> M =
      parseAssembly(Context, "define void @a() {\n"
                             "entry:\n"
                             "  call void @b()\n"
                             "  call void @e()\n"
                             "  ret void\n"
                             "}\n"
                             "define void @b() {\n"
                             "entry:\n"
                             "  call void @c()\n"
                             "  call void @d()\n"
                             "  ret void\n"
                             "}\n"
                             "define void @c() {\n"
                             "entry:\n"
                             "  call void @d()\n"
                             "  call void @g()\n"
                             "  ret void\n"
                             "}\n"
                             "define void @d() {\n"
                             "entry:\n"
                             "  call void @e()\n"
                             "  call void @f()\n"
                             "  ret void\n"
                             "}\n"
                             "define void @e() {\n"
                             "entry:\n"
                             "  call void @f()\n"
                             "  ret void\n"
                             "}\n"
                             "define void @f() {\n"
                             "entry:\n"
                             "  store void()* @b, void()** undef\n"
                             "  call void @g()\n"
                             "  ret void\n"
                             "}\n"
                             "define void @g() {\n"
                             "entry:\n"
                             "  store void()* @a, void()** undef\n"
                             "  ret void\n"
                             "}\n");
  LazyCallGraph CG = buildCG(*M);

  // Force the graph to be fully expanded.
  CG.buildRefSCCs();
  auto I = CG.postorder_ref_scc_begin();
  LazyCallGraph::RefSCC &RC = *I++;
  EXPECT_EQ(CG.postorder_ref_scc_end(), I);

  LazyCallGraph::Node &A = *CG.lookup(lookupFunction(*M, "a"));
  LazyCallGraph::Node &B = *CG.lookup(lookupFunction(*M, "b"));
  LazyCallGraph::Node &C = *CG.lookup(lookupFunction(*M, "c"));
  LazyCallGraph::Node &D = *CG.lookup(lookupFunction(*M, "d"));
  LazyCallGraph::Node &E = *CG.lookup(lookupFunction(*M, "e"));
  LazyCallGraph::Node &F = *CG.lookup(lookupFunction(*M, "f"));
  LazyCallGraph::Node &G = *CG.lookup(lookupFunction(*M, "g"));
  LazyCallGraph::SCC &AC = *CG.lookupSCC(A);
  LazyCallGraph::SCC &BC = *CG.lookupSCC(B);
  LazyCallGraph::SCC &CC = *CG.lookupSCC(C);
  LazyCallGraph::SCC &DC = *CG.lookupSCC(D);
  LazyCallGraph::SCC &EC = *CG.lookupSCC(E);
  LazyCallGraph::SCC &FC = *CG.lookupSCC(F);
  LazyCallGraph::SCC &GC = *CG.lookupSCC(G);

  // Remove the extra edges that were used to force a particular post-order.
  RC.switchTrivialInternalEdgeToRef(C, D);
  RC.switchTrivialInternalEdgeToRef(D, E);

  // Check the initial post-order. We ensure this order with the extra edges
  // that are nuked above.
  ASSERT_EQ(7, RC.size());
  EXPECT_EQ(&GC, &RC[0]);
  EXPECT_EQ(&FC, &RC[1]);
  EXPECT_EQ(&EC, &RC[2]);
  EXPECT_EQ(&DC, &RC[3]);
  EXPECT_EQ(&CC, &RC[4]);
  EXPECT_EQ(&BC, &RC[5]);
  EXPECT_EQ(&AC, &RC[6]);

  // Switch F -> B to a call edge. This merges B, D, and F into a single SCC,
  // and has to place the C and E SCCs on either side of it:
  //   A          A    |
  //  / \        / \   |
  // B   E      |   E  |
  // |\  |       \ /   |
  // | D |  ->    B    |
  // |  \|       / \   |
  // C   F      C   |  |
  //  \ /        \ /   |
  //   G          G    |
  EXPECT_TRUE(RC.switchInternalEdgeToCall(
      F, B, [&](ArrayRef<LazyCallGraph::SCC *> MergedCs) {
        ASSERT_EQ(2u, MergedCs.size());
        EXPECT_EQ(&FC, MergedCs[0]);
        EXPECT_EQ(&DC, MergedCs[1]);
      }));
  EXPECT_EQ(3, BC.size());

  // And make sure the postorder was updated.
  ASSERT_EQ(5, RC.size());
  EXPECT_EQ(&GC, &RC[0]);
  EXPECT_EQ(&CC, &RC[1]);
  EXPECT_EQ(&BC, &RC[2]);
  EXPECT_EQ(&EC, &RC[3]);
  EXPECT_EQ(&AC, &RC[4]);
}

// Test for IR containing constants using blockaddress constant expressions.
// These are truly unique constructs: constant expressions with non-constant
// operands.
TEST(LazyCallGraphTest, HandleBlockAddress) {
  LLVMContext Context;
  std::unique_ptr<Module> M =
      parseAssembly(Context, "define void @f() {\n"
                             "entry:\n"
                             "  ret void\n"
                             "bb:\n"
                             "  unreachable\n"
                             "}\n"
                             "define void @g(i8** %ptr) {\n"
                             "entry:\n"
                             "  store i8* blockaddress(@f, %bb), i8** %ptr\n"
                             "  ret void\n"
                             "}\n");
  LazyCallGraph CG = buildCG(*M);

  CG.buildRefSCCs();
  auto I = CG.postorder_ref_scc_begin();
  LazyCallGraph::RefSCC &FRC = *I++;
  LazyCallGraph::RefSCC &GRC = *I++;
  EXPECT_EQ(CG.postorder_ref_scc_end(), I);

  LazyCallGraph::Node &F = *CG.lookup(lookupFunction(*M, "f"));
  LazyCallGraph::Node &G = *CG.lookup(lookupFunction(*M, "g"));
  EXPECT_EQ(&FRC, CG.lookupRefSCC(F));
  EXPECT_EQ(&GRC, CG.lookupRefSCC(G));
  EXPECT_TRUE(GRC.isParentOf(FRC));
}

TEST(LazyCallGraphTest, ReplaceNodeFunction) {
  LLVMContext Context;
  // A graph with several different kinds of edges pointing at a particular
  // function.
  std::unique_ptr<Module> M =
      parseAssembly(Context,
                    "define void @a(i8** %ptr) {\n"
                    "entry:\n"
                    "  store i8* bitcast (void(i8**)* @d to i8*), i8** %ptr\n"
                    "  ret void\n"
                    "}\n"
                    "define void @b(i8** %ptr) {\n"
                    "entry:\n"
                    "  store i8* bitcast (void(i8**)* @d to i8*), i8** %ptr\n"
                    "  store i8* bitcast (void(i8**)* @d to i8*), i8** %ptr\n"
                    "  call void @d(i8** %ptr)"
                    "  ret void\n"
                    "}\n"
                    "define void @c(i8** %ptr) {\n"
                    "entry:\n"
                    "  call void @d(i8** %ptr)"
                    "  call void @d(i8** %ptr)"
                    "  store i8* bitcast (void(i8**)* @d to i8*), i8** %ptr\n"
                    "  ret void\n"
                    "}\n"
                    "define void @d(i8** %ptr) {\n"
                    "entry:\n"
                    "  store i8* bitcast (void(i8**)* @b to i8*), i8** %ptr\n"
                    "  call void @c(i8** %ptr)"
                    "  call void @d(i8** %ptr)"
                    "  store i8* bitcast (void(i8**)* @d to i8*), i8** %ptr\n"
                    "  ret void\n"
                    "}\n");
  LazyCallGraph CG = buildCG(*M);

  // Force the graph to be fully expanded.
  CG.buildRefSCCs();
  auto I = CG.postorder_ref_scc_begin();
  LazyCallGraph::RefSCC &RC1 = *I++;
  LazyCallGraph::RefSCC &RC2 = *I++;
  EXPECT_EQ(CG.postorder_ref_scc_end(), I);

  ASSERT_EQ(2, RC1.size());
  LazyCallGraph::SCC &C1 = RC1[0];
  LazyCallGraph::SCC &C2 = RC1[1];

  LazyCallGraph::Node &AN = *CG.lookup(lookupFunction(*M, "a"));
  LazyCallGraph::Node &BN = *CG.lookup(lookupFunction(*M, "b"));
  LazyCallGraph::Node &CN = *CG.lookup(lookupFunction(*M, "c"));
  LazyCallGraph::Node &DN = *CG.lookup(lookupFunction(*M, "d"));
  EXPECT_EQ(&C1, CG.lookupSCC(DN));
  EXPECT_EQ(&C1, CG.lookupSCC(CN));
  EXPECT_EQ(&C2, CG.lookupSCC(BN));
  EXPECT_EQ(&RC1, CG.lookupRefSCC(DN));
  EXPECT_EQ(&RC1, CG.lookupRefSCC(CN));
  EXPECT_EQ(&RC1, CG.lookupRefSCC(BN));
  EXPECT_EQ(&RC2, CG.lookupRefSCC(AN));

  // Now we need to build a new function 'e' with the same signature as 'd'.
  Function &D = DN.getFunction();
  Function &E = *Function::Create(D.getFunctionType(), D.getLinkage(), "e");
  D.getParent()->getFunctionList().insert(D.getIterator(), &E);

  // Change each use of 'd' to use 'e'. This is particularly easy as they have
  // the same type.
  D.replaceAllUsesWith(&E);

  // Splice the body of the old function into the new one.
  E.getBasicBlockList().splice(E.begin(), D.getBasicBlockList());
  // And fix up the one argument.
  D.arg_begin()->replaceAllUsesWith(&*E.arg_begin());
  E.arg_begin()->takeName(&*D.arg_begin());

  // Now replace the function in the graph.
  RC1.replaceNodeFunction(DN, E);

  EXPECT_EQ(&E, &DN.getFunction());
  EXPECT_EQ(&DN, &(*CN)[DN].getNode());
  EXPECT_EQ(&DN, &(*BN)[DN].getNode());
}

TEST(LazyCallGraphTest, RemoveFunctionWithSpurriousRef) {
  LLVMContext Context;
  // A graph with a couple of RefSCCs.
  std::unique_ptr<Module> M =
      parseAssembly(Context,
                    "define void @a(i8** %ptr) {\n"
                    "entry:\n"
                    "  store i8* bitcast (void(i8**)* @d to i8*), i8** %ptr\n"
                    "  ret void\n"
                    "}\n"
                    "define void @b(i8** %ptr) {\n"
                    "entry:\n"
                    "  store i8* bitcast (void(i8**)* @c to i8*), i8** %ptr\n"
                    "  ret void\n"
                    "}\n"
                    "define void @c(i8** %ptr) {\n"
                    "entry:\n"
                    "  call void @d(i8** %ptr)"
                    "  ret void\n"
                    "}\n"
                    "define void @d(i8** %ptr) {\n"
                    "entry:\n"
                    "  call void @c(i8** %ptr)"
                    "  store i8* bitcast (void(i8**)* @b to i8*), i8** %ptr\n"
                    "  ret void\n"
                    "}\n"
                    "define void @dead() {\n"
                    "entry:\n"
                    "  ret void\n"
                    "}\n");
  LazyCallGraph CG = buildCG(*M);

  // Insert spurious ref edges.
  LazyCallGraph::Node &AN = CG.get(lookupFunction(*M, "a"));
  LazyCallGraph::Node &BN = CG.get(lookupFunction(*M, "b"));
  LazyCallGraph::Node &CN = CG.get(lookupFunction(*M, "c"));
  LazyCallGraph::Node &DN = CG.get(lookupFunction(*M, "d"));
  LazyCallGraph::Node &DeadN = CG.get(lookupFunction(*M, "dead"));
  AN.populate();
  BN.populate();
  CN.populate();
  DN.populate();
  DeadN.populate();
  CG.insertEdge(AN, DeadN, LazyCallGraph::Edge::Ref);
  CG.insertEdge(BN, DeadN, LazyCallGraph::Edge::Ref);
  CG.insertEdge(CN, DeadN, LazyCallGraph::Edge::Ref);
  CG.insertEdge(DN, DeadN, LazyCallGraph::Edge::Ref);

  // Force the graph to be fully expanded.
  CG.buildRefSCCs();
  auto I = CG.postorder_ref_scc_begin();
  LazyCallGraph::RefSCC &DeadRC = *I++;
  LazyCallGraph::RefSCC &RC1 = *I++;
  LazyCallGraph::RefSCC &RC2 = *I++;
  EXPECT_EQ(CG.postorder_ref_scc_end(), I);

  ASSERT_EQ(2, RC1.size());
  LazyCallGraph::SCC &C1 = RC1[0];
  LazyCallGraph::SCC &C2 = RC1[1];

  EXPECT_EQ(&DeadRC, CG.lookupRefSCC(DeadN));
  EXPECT_EQ(&C1, CG.lookupSCC(DN));
  EXPECT_EQ(&C1, CG.lookupSCC(CN));
  EXPECT_EQ(&C2, CG.lookupSCC(BN));
  EXPECT_EQ(&RC1, CG.lookupRefSCC(DN));
  EXPECT_EQ(&RC1, CG.lookupRefSCC(CN));
  EXPECT_EQ(&RC1, CG.lookupRefSCC(BN));
  EXPECT_EQ(&RC2, CG.lookupRefSCC(AN));

  // Now delete 'dead'. There are no uses of this function but there are
  // spurious references.
  CG.removeDeadFunction(DeadN.getFunction());

  // The only observable change should be that the RefSCC is gone from the
  // postorder sequence.
  I = CG.postorder_ref_scc_begin();
  EXPECT_EQ(&RC1, &*I++);
  EXPECT_EQ(&RC2, &*I++);
  EXPECT_EQ(CG.postorder_ref_scc_end(), I);
}
}
