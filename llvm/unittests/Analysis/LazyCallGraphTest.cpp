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
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;

namespace {

std::unique_ptr<Module> parseAssembly(const char *Assembly) {
  auto M = make_unique<Module>("Module", getGlobalContext());

  SMDiagnostic Error;
  bool Parsed =
      ParseAssemblyString(Assembly, M.get(), Error, M->getContext()) == M.get();

  std::string ErrMsg;
  raw_string_ostream OS(ErrMsg);
  Error.print("", OS);

  // A failure here means that the test itself is buggy.
  if (!Parsed)
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

TEST(LazyCallGraphTest, BasicGraphFormation) {
  std::unique_ptr<Module> M = parseAssembly(DiamondOfTriangles);
  LazyCallGraph CG(*M);

  // The order of the entry nodes should be stable w.r.t. the source order of
  // the IR, and everything in our module is an entry node, so just directly
  // build variables for each node.
  auto I = CG.begin();
  LazyCallGraph::Node &A1 = *I++;
  EXPECT_EQ("a1", A1.getFunction().getName());
  LazyCallGraph::Node &A2 = *I++;
  EXPECT_EQ("a2", A2.getFunction().getName());
  LazyCallGraph::Node &A3 = *I++;
  EXPECT_EQ("a3", A3.getFunction().getName());
  LazyCallGraph::Node &B1 = *I++;
  EXPECT_EQ("b1", B1.getFunction().getName());
  LazyCallGraph::Node &B2 = *I++;
  EXPECT_EQ("b2", B2.getFunction().getName());
  LazyCallGraph::Node &B3 = *I++;
  EXPECT_EQ("b3", B3.getFunction().getName());
  LazyCallGraph::Node &C1 = *I++;
  EXPECT_EQ("c1", C1.getFunction().getName());
  LazyCallGraph::Node &C2 = *I++;
  EXPECT_EQ("c2", C2.getFunction().getName());
  LazyCallGraph::Node &C3 = *I++;
  EXPECT_EQ("c3", C3.getFunction().getName());
  LazyCallGraph::Node &D1 = *I++;
  EXPECT_EQ("d1", D1.getFunction().getName());
  LazyCallGraph::Node &D2 = *I++;
  EXPECT_EQ("d2", D2.getFunction().getName());
  LazyCallGraph::Node &D3 = *I++;
  EXPECT_EQ("d3", D3.getFunction().getName());
  EXPECT_EQ(CG.end(), I);

  // Build vectors and sort them for the rest of the assertions to make them
  // independent of order.
  std::vector<std::string> Nodes;

  for (LazyCallGraph::Node &N : A1)
    Nodes.push_back(N.getFunction().getName());
  std::sort(Nodes.begin(), Nodes.end());
  EXPECT_EQ("a2", Nodes[0]);
  EXPECT_EQ("b2", Nodes[1]);
  EXPECT_EQ("c3", Nodes[2]);
  Nodes.clear();

  EXPECT_EQ(A2.end(), std::next(A2.begin()));
  EXPECT_EQ("a3", A2.begin()->getFunction().getName());
  EXPECT_EQ(A3.end(), std::next(A3.begin()));
  EXPECT_EQ("a1", A3.begin()->getFunction().getName());

  for (LazyCallGraph::Node &N : B1)
    Nodes.push_back(N.getFunction().getName());
  std::sort(Nodes.begin(), Nodes.end());
  EXPECT_EQ("b2", Nodes[0]);
  EXPECT_EQ("d3", Nodes[1]);
  Nodes.clear();

  EXPECT_EQ(B2.end(), std::next(B2.begin()));
  EXPECT_EQ("b3", B2.begin()->getFunction().getName());
  EXPECT_EQ(B3.end(), std::next(B3.begin()));
  EXPECT_EQ("b1", B3.begin()->getFunction().getName());

  for (LazyCallGraph::Node &N : C1)
    Nodes.push_back(N.getFunction().getName());
  std::sort(Nodes.begin(), Nodes.end());
  EXPECT_EQ("c2", Nodes[0]);
  EXPECT_EQ("d2", Nodes[1]);
  Nodes.clear();

  EXPECT_EQ(C2.end(), std::next(C2.begin()));
  EXPECT_EQ("c3", C2.begin()->getFunction().getName());
  EXPECT_EQ(C3.end(), std::next(C3.begin()));
  EXPECT_EQ("c1", C3.begin()->getFunction().getName());

  EXPECT_EQ(D1.end(), std::next(D1.begin()));
  EXPECT_EQ("d2", D1.begin()->getFunction().getName());
  EXPECT_EQ(D2.end(), std::next(D2.begin()));
  EXPECT_EQ("d3", D2.begin()->getFunction().getName());
  EXPECT_EQ(D3.end(), std::next(D3.begin()));
  EXPECT_EQ("d1", D3.begin()->getFunction().getName());

  // Now lets look at the SCCs.
  auto SCCI = CG.postorder_scc_begin();

  LazyCallGraph::SCC &D = *SCCI++;
  for (LazyCallGraph::Node *N : D)
    Nodes.push_back(N->getFunction().getName());
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

  LazyCallGraph::SCC &C = *SCCI++;
  for (LazyCallGraph::Node *N : C)
    Nodes.push_back(N->getFunction().getName());
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

  LazyCallGraph::SCC &B = *SCCI++;
  for (LazyCallGraph::Node *N : B)
    Nodes.push_back(N->getFunction().getName());
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

  LazyCallGraph::SCC &A = *SCCI++;
  for (LazyCallGraph::Node *N : A)
    Nodes.push_back(N->getFunction().getName());
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

  EXPECT_EQ(CG.postorder_scc_end(), SCCI);
}

static Function &lookupFunction(Module &M, StringRef Name) {
  for (Function &F : M)
    if (F.getName() == Name)
      return F;
  report_fatal_error("Couldn't find function!");
}

TEST(LazyCallGraphTest, BasicGraphMutation) {
  std::unique_ptr<Module> M = parseAssembly(
      "define void @a() {\n"
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
  LazyCallGraph CG(*M);

  LazyCallGraph::Node &A = CG.get(lookupFunction(*M, "a"));
  LazyCallGraph::Node &B = CG.get(lookupFunction(*M, "b"));
  EXPECT_EQ(2, std::distance(A.begin(), A.end()));
  EXPECT_EQ(0, std::distance(B.begin(), B.end()));

  CG.insertEdge(B, lookupFunction(*M, "c"));
  EXPECT_EQ(1, std::distance(B.begin(), B.end()));
  LazyCallGraph::Node &C = *B.begin();
  EXPECT_EQ(0, std::distance(C.begin(), C.end()));

  CG.insertEdge(C, B.getFunction());
  EXPECT_EQ(1, std::distance(C.begin(), C.end()));
  EXPECT_EQ(&B, &*C.begin());

  CG.insertEdge(C, C.getFunction());
  EXPECT_EQ(2, std::distance(C.begin(), C.end()));
  EXPECT_EQ(&B, &*C.begin());
  EXPECT_EQ(&C, &*std::next(C.begin()));

  CG.removeEdge(C, B.getFunction());
  EXPECT_EQ(1, std::distance(C.begin(), C.end()));
  EXPECT_EQ(&C, &*C.begin());

  CG.removeEdge(C, C.getFunction());
  EXPECT_EQ(0, std::distance(C.begin(), C.end()));

  CG.removeEdge(B, C.getFunction());
  EXPECT_EQ(0, std::distance(B.begin(), B.end()));
}

TEST(LazyCallGraphTest, MultiArmSCC) {
  // Two interlocking cycles. The really useful thing about this SCC is that it
  // will require Tarjan's DFS to backtrack and finish processing all of the
  // children of each node in the SCC.
  std::unique_ptr<Module> M = parseAssembly(
      "define void @a() {\n"
      "entry:\n"
      "  call void @b()\n"
      "  call void @d()\n"
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
      "}\n"
      "define void @d() {\n"
      "entry:\n"
      "  call void @e()\n"
      "  ret void\n"
      "}\n"
      "define void @e() {\n"
      "entry:\n"
      "  call void @a()\n"
      "  ret void\n"
      "}\n");
  LazyCallGraph CG(*M);

  // Force the graph to be fully expanded.
  auto SCCI = CG.postorder_scc_begin();
  LazyCallGraph::SCC &SCC = *SCCI++;
  EXPECT_EQ(CG.postorder_scc_end(), SCCI);

  LazyCallGraph::Node &A = *CG.lookup(lookupFunction(*M, "a"));
  LazyCallGraph::Node &B = *CG.lookup(lookupFunction(*M, "b"));
  LazyCallGraph::Node &C = *CG.lookup(lookupFunction(*M, "c"));
  LazyCallGraph::Node &D = *CG.lookup(lookupFunction(*M, "d"));
  LazyCallGraph::Node &E = *CG.lookup(lookupFunction(*M, "e"));
  EXPECT_EQ(&SCC, CG.lookupSCC(A));
  EXPECT_EQ(&SCC, CG.lookupSCC(B));
  EXPECT_EQ(&SCC, CG.lookupSCC(C));
  EXPECT_EQ(&SCC, CG.lookupSCC(D));
  EXPECT_EQ(&SCC, CG.lookupSCC(E));
}

TEST(LazyCallGraphTest, OutgoingSCCEdgeInsertion) {
  std::unique_ptr<Module> M = parseAssembly(
      "define void @a() {\n"
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
  LazyCallGraph CG(*M);

  // Force the graph to be fully expanded.
  for (LazyCallGraph::SCC &C : CG.postorder_sccs())
    (void)C;

  LazyCallGraph::Node &A = *CG.lookup(lookupFunction(*M, "a"));
  LazyCallGraph::Node &B = *CG.lookup(lookupFunction(*M, "b"));
  LazyCallGraph::Node &C = *CG.lookup(lookupFunction(*M, "c"));
  LazyCallGraph::Node &D = *CG.lookup(lookupFunction(*M, "d"));
  LazyCallGraph::SCC &AC = *CG.lookupSCC(A);
  LazyCallGraph::SCC &BC = *CG.lookupSCC(B);
  LazyCallGraph::SCC &CC = *CG.lookupSCC(C);
  LazyCallGraph::SCC &DC = *CG.lookupSCC(D);
  EXPECT_TRUE(AC.isAncestorOf(BC));
  EXPECT_TRUE(AC.isAncestorOf(CC));
  EXPECT_TRUE(AC.isAncestorOf(DC));
  EXPECT_TRUE(DC.isDescendantOf(AC));
  EXPECT_TRUE(DC.isDescendantOf(BC));
  EXPECT_TRUE(DC.isDescendantOf(CC));

  EXPECT_EQ(2, std::distance(A.begin(), A.end()));
  AC.insertOutgoingEdge(A, D);
  EXPECT_EQ(3, std::distance(A.begin(), A.end()));
  EXPECT_TRUE(AC.isParentOf(DC));
  EXPECT_EQ(&AC, CG.lookupSCC(A));
  EXPECT_EQ(&BC, CG.lookupSCC(B));
  EXPECT_EQ(&CC, CG.lookupSCC(C));
  EXPECT_EQ(&DC, CG.lookupSCC(D));
}

TEST(LazyCallGraphTest, IncomingSCCEdgeInsertion) {
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
  std::unique_ptr<Module> M = parseAssembly(DiamondOfTriangles);
  LazyCallGraph CG(*M);

  // Force the graph to be fully expanded.
  for (LazyCallGraph::SCC &C : CG.postorder_sccs())
    (void)C;

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
  LazyCallGraph::SCC &AC = *CG.lookupSCC(A1);
  LazyCallGraph::SCC &BC = *CG.lookupSCC(B1);
  LazyCallGraph::SCC &CC = *CG.lookupSCC(C1);
  LazyCallGraph::SCC &DC = *CG.lookupSCC(D1);
  ASSERT_EQ(&AC, CG.lookupSCC(A2));
  ASSERT_EQ(&AC, CG.lookupSCC(A3));
  ASSERT_EQ(&BC, CG.lookupSCC(B2));
  ASSERT_EQ(&BC, CG.lookupSCC(B3));
  ASSERT_EQ(&CC, CG.lookupSCC(C2));
  ASSERT_EQ(&CC, CG.lookupSCC(C3));
  ASSERT_EQ(&DC, CG.lookupSCC(D2));
  ASSERT_EQ(&DC, CG.lookupSCC(D3));
  ASSERT_EQ(1, std::distance(D2.begin(), D2.end()));

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
  CC.insertIncomingEdge(D2, C2);
  // Make sure we connected the nodes.
  EXPECT_EQ(2, std::distance(D2.begin(), D2.end()));

  // Make sure we have the correct nodes in the SCC sets.
  EXPECT_EQ(&AC, CG.lookupSCC(A1));
  EXPECT_EQ(&AC, CG.lookupSCC(A2));
  EXPECT_EQ(&AC, CG.lookupSCC(A3));
  EXPECT_EQ(&BC, CG.lookupSCC(B1));
  EXPECT_EQ(&BC, CG.lookupSCC(B2));
  EXPECT_EQ(&BC, CG.lookupSCC(B3));
  EXPECT_EQ(&CC, CG.lookupSCC(C1));
  EXPECT_EQ(&CC, CG.lookupSCC(C2));
  EXPECT_EQ(&CC, CG.lookupSCC(C3));
  EXPECT_EQ(&CC, CG.lookupSCC(D1));
  EXPECT_EQ(&CC, CG.lookupSCC(D2));
  EXPECT_EQ(&CC, CG.lookupSCC(D3));

  // And that ancestry tests have been updated.
  EXPECT_TRUE(AC.isParentOf(BC));
  EXPECT_TRUE(AC.isParentOf(CC));
  EXPECT_FALSE(AC.isAncestorOf(DC));
  EXPECT_FALSE(BC.isAncestorOf(DC));
  EXPECT_FALSE(CC.isAncestorOf(DC));
}

TEST(LazyCallGraphTest, IncomingSCCEdgeInsertionMidTraversal) {
  // This is the same fundamental test as the previous, but we perform it
  // having only partially walked the SCCs of the graph.
  std::unique_ptr<Module> M = parseAssembly(DiamondOfTriangles);
  LazyCallGraph CG(*M);

  // Walk the SCCs until we find the one containing 'c1'.
  auto SCCI = CG.postorder_scc_begin(), SCCE = CG.postorder_scc_end();
  ASSERT_NE(SCCI, SCCE);
  LazyCallGraph::SCC &DC = *SCCI;
  ASSERT_NE(&DC, nullptr);
  ++SCCI;
  ASSERT_NE(SCCI, SCCE);
  LazyCallGraph::SCC &CC = *SCCI;
  ASSERT_NE(&CC, nullptr);

  ASSERT_EQ(nullptr, CG.lookup(lookupFunction(*M, "a1")));
  ASSERT_EQ(nullptr, CG.lookup(lookupFunction(*M, "a2")));
  ASSERT_EQ(nullptr, CG.lookup(lookupFunction(*M, "a3")));
  ASSERT_EQ(nullptr, CG.lookup(lookupFunction(*M, "b1")));
  ASSERT_EQ(nullptr, CG.lookup(lookupFunction(*M, "b2")));
  ASSERT_EQ(nullptr, CG.lookup(lookupFunction(*M, "b3")));
  LazyCallGraph::Node &C1 = *CG.lookup(lookupFunction(*M, "c1"));
  LazyCallGraph::Node &C2 = *CG.lookup(lookupFunction(*M, "c2"));
  LazyCallGraph::Node &C3 = *CG.lookup(lookupFunction(*M, "c3"));
  LazyCallGraph::Node &D1 = *CG.lookup(lookupFunction(*M, "d1"));
  LazyCallGraph::Node &D2 = *CG.lookup(lookupFunction(*M, "d2"));
  LazyCallGraph::Node &D3 = *CG.lookup(lookupFunction(*M, "d3"));
  ASSERT_EQ(&CC, CG.lookupSCC(C1));
  ASSERT_EQ(&CC, CG.lookupSCC(C2));
  ASSERT_EQ(&CC, CG.lookupSCC(C3));
  ASSERT_EQ(&DC, CG.lookupSCC(D1));
  ASSERT_EQ(&DC, CG.lookupSCC(D2));
  ASSERT_EQ(&DC, CG.lookupSCC(D3));
  ASSERT_EQ(1, std::distance(D2.begin(), D2.end()));

  CC.insertIncomingEdge(D2, C2);
  EXPECT_EQ(2, std::distance(D2.begin(), D2.end()));

  // Make sure we have the correct nodes in the SCC sets.
  EXPECT_EQ(&CC, CG.lookupSCC(C1));
  EXPECT_EQ(&CC, CG.lookupSCC(C2));
  EXPECT_EQ(&CC, CG.lookupSCC(C3));
  EXPECT_EQ(&CC, CG.lookupSCC(D1));
  EXPECT_EQ(&CC, CG.lookupSCC(D2));
  EXPECT_EQ(&CC, CG.lookupSCC(D3));

  // Check that we can form the last two SCCs now in a coherent way.
  ++SCCI;
  EXPECT_NE(SCCI, SCCE);
  LazyCallGraph::SCC &BC = *SCCI;
  EXPECT_NE(&BC, nullptr);
  EXPECT_EQ(&BC, CG.lookupSCC(*CG.lookup(lookupFunction(*M, "b1"))));
  EXPECT_EQ(&BC, CG.lookupSCC(*CG.lookup(lookupFunction(*M, "b2"))));
  EXPECT_EQ(&BC, CG.lookupSCC(*CG.lookup(lookupFunction(*M, "b3"))));
  ++SCCI;
  EXPECT_NE(SCCI, SCCE);
  LazyCallGraph::SCC &AC = *SCCI;
  EXPECT_NE(&AC, nullptr);
  EXPECT_EQ(&AC, CG.lookupSCC(*CG.lookup(lookupFunction(*M, "a1"))));
  EXPECT_EQ(&AC, CG.lookupSCC(*CG.lookup(lookupFunction(*M, "a2"))));
  EXPECT_EQ(&AC, CG.lookupSCC(*CG.lookup(lookupFunction(*M, "a3"))));
  ++SCCI;
  EXPECT_EQ(SCCI, SCCE);
}

TEST(LazyCallGraphTest, InterSCCEdgeRemoval) {
  std::unique_ptr<Module> M = parseAssembly(
      "define void @a() {\n"
      "entry:\n"
      "  call void @b()\n"
      "  ret void\n"
      "}\n"
      "define void @b() {\n"
      "entry:\n"
      "  ret void\n"
      "}\n");
  LazyCallGraph CG(*M);

  // Force the graph to be fully expanded.
  for (LazyCallGraph::SCC &C : CG.postorder_sccs())
    (void)C;

  LazyCallGraph::Node &A = *CG.lookup(lookupFunction(*M, "a"));
  LazyCallGraph::Node &B = *CG.lookup(lookupFunction(*M, "b"));
  LazyCallGraph::SCC &AC = *CG.lookupSCC(A);
  LazyCallGraph::SCC &BC = *CG.lookupSCC(B);

  EXPECT_EQ("b", A.begin()->getFunction().getName());
  EXPECT_EQ(B.end(), B.begin());
  EXPECT_EQ(&AC, &*BC.parent_begin());

  AC.removeInterSCCEdge(A, B);

  EXPECT_EQ(A.end(), A.begin());
  EXPECT_EQ(B.end(), B.begin());
  EXPECT_EQ(BC.parent_end(), BC.parent_begin());
}

TEST(LazyCallGraphTest, IntraSCCEdgeInsertion) {
  std::unique_ptr<Module> M1 = parseAssembly(
      "define void @a() {\n"
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
  LazyCallGraph CG1(*M1);

  // Force the graph to be fully expanded.
  auto SCCI = CG1.postorder_scc_begin();
  LazyCallGraph::SCC &SCC = *SCCI++;
  EXPECT_EQ(CG1.postorder_scc_end(), SCCI);

  LazyCallGraph::Node &A = *CG1.lookup(lookupFunction(*M1, "a"));
  LazyCallGraph::Node &B = *CG1.lookup(lookupFunction(*M1, "b"));
  LazyCallGraph::Node &C = *CG1.lookup(lookupFunction(*M1, "c"));
  EXPECT_EQ(&SCC, CG1.lookupSCC(A));
  EXPECT_EQ(&SCC, CG1.lookupSCC(B));
  EXPECT_EQ(&SCC, CG1.lookupSCC(C));

  // Insert an edge from 'a' to 'c'. Nothing changes about the SCCs.
  SCC.insertIntraSCCEdge(A, C);
  EXPECT_EQ(2, std::distance(A.begin(), A.end()));
  EXPECT_EQ(&SCC, CG1.lookupSCC(A));
  EXPECT_EQ(&SCC, CG1.lookupSCC(B));
  EXPECT_EQ(&SCC, CG1.lookupSCC(C));

  // Insert a self edge from 'a' back to 'a'.
  SCC.insertIntraSCCEdge(A, A);
  EXPECT_EQ(3, std::distance(A.begin(), A.end()));
  EXPECT_EQ(&SCC, CG1.lookupSCC(A));
  EXPECT_EQ(&SCC, CG1.lookupSCC(B));
  EXPECT_EQ(&SCC, CG1.lookupSCC(C));
}

TEST(LazyCallGraphTest, IntraSCCEdgeRemoval) {
  // A nice fully connected (including self-edges) SCC.
  std::unique_ptr<Module> M1 = parseAssembly(
      "define void @a() {\n"
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
  LazyCallGraph CG1(*M1);

  // Force the graph to be fully expanded.
  auto SCCI = CG1.postorder_scc_begin();
  LazyCallGraph::SCC &SCC = *SCCI++;
  EXPECT_EQ(CG1.postorder_scc_end(), SCCI);

  LazyCallGraph::Node &A = *CG1.lookup(lookupFunction(*M1, "a"));
  LazyCallGraph::Node &B = *CG1.lookup(lookupFunction(*M1, "b"));
  LazyCallGraph::Node &C = *CG1.lookup(lookupFunction(*M1, "c"));
  EXPECT_EQ(&SCC, CG1.lookupSCC(A));
  EXPECT_EQ(&SCC, CG1.lookupSCC(B));
  EXPECT_EQ(&SCC, CG1.lookupSCC(C));

  // Remove the edge from b -> a, which should leave the 3 functions still in
  // a single connected component because of a -> b -> c -> a.
  SmallVector<LazyCallGraph::SCC *, 1> NewSCCs = SCC.removeIntraSCCEdge(B, A);
  EXPECT_EQ(0u, NewSCCs.size());
  EXPECT_EQ(&SCC, CG1.lookupSCC(A));
  EXPECT_EQ(&SCC, CG1.lookupSCC(B));
  EXPECT_EQ(&SCC, CG1.lookupSCC(C));

  // Remove the edge from c -> a, which should leave 'a' in the original SCC
  // and form a new SCC for 'b' and 'c'.
  NewSCCs = SCC.removeIntraSCCEdge(C, A);
  EXPECT_EQ(1u, NewSCCs.size());
  EXPECT_EQ(&SCC, CG1.lookupSCC(A));
  EXPECT_EQ(1, std::distance(SCC.begin(), SCC.end()));
  LazyCallGraph::SCC *SCC2 = CG1.lookupSCC(B);
  EXPECT_EQ(SCC2, CG1.lookupSCC(C));
  EXPECT_EQ(SCC2, NewSCCs[0]);
}

}
