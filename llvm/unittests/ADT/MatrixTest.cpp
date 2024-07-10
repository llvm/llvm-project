#include "llvm/ADT/Matrix.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "gtest/gtest.h"

using namespace llvm;

template <typename T> static SmallVector<T> getDummyValues(size_t NValues) {
  SmallVector<T> Ret(NValues);
  for (size_t Idx = 0; Idx < NValues; ++Idx)
    Ret[Idx] = T(Idx + 1);
  return Ret;
}

template <typename T> class MatrixTest : public testing::Test {
protected:
  MatrixTest()
      : ColumnInitMatrix(8), SmallMatrix(2, 2), OtherSmall(3, 3),
        LargeMatrix(16, 16) {}
  MatrixStorage<T, 16> ColumnInitMatrix;
  MatrixStorage<T> SmallMatrix;
  MatrixStorage<T> OtherSmall;
  MatrixStorage<T> LargeMatrix;
  SmallVector<T> getDummyRow(size_t NValues) {
    return getDummyValues<T>(NValues);
  }
};

using MatrixTestTypes = ::testing::Types<int64_t, DynamicAPInt>;
TYPED_TEST_SUITE(MatrixTest, MatrixTestTypes, );

TYPED_TEST(MatrixTest, Construction) {
  auto &E = this->ColumnInitMatrix;
  ASSERT_TRUE(E.empty());
  EXPECT_EQ(E.getNumCols(), 8u);
  E.setNumCols(3);
  EXPECT_EQ(E.getNumCols(), 3u);
  EXPECT_TRUE(E.empty());
  EXPECT_EQ(E.getNumRows(), 0u);
  auto &M = this->SmallMatrix;
  EXPECT_FALSE(M.empty());
  EXPECT_EQ(M.size(), 4u);
  EXPECT_EQ(M.getNumRows(), 2u);
  EXPECT_EQ(M.getNumCols(), 2u);
}

TYPED_TEST(MatrixTest, CopyConstruction) {
  auto &OldMat = this->SmallMatrix;
  auto V = JaggedArrayView<TypeParam>{OldMat};
  V[0] = this->getDummyRow(2);
  V[0].pop_back();
  V[1] = this->getDummyRow(2);
  V[1].pop_front();
  ASSERT_EQ(V.getRowSpan(), 2u);
  ASSERT_EQ(V.getColSpan(0), 1u);
  ASSERT_EQ(V.getColSpan(1), 1u);
  EXPECT_EQ(V[0][0], 1);
  EXPECT_EQ(V[1][0], 2);
  EXPECT_EQ(JaggedArrayView<TypeParam>{OldMat}[0][0], 1);
  EXPECT_EQ(JaggedArrayView<TypeParam>{OldMat}[0][1], 2);
  EXPECT_EQ(JaggedArrayView<TypeParam>{OldMat}[1][0], 1);
  EXPECT_EQ(JaggedArrayView<TypeParam>{OldMat}[1][1], 2);
  MatrixStorage<TypeParam> NewMat{OldMat};
  JaggedArrayView<TypeParam> C{V, NewMat};
  ASSERT_EQ(C.getRowSpan(), 2u);
  ASSERT_EQ(C.getColSpan(0), 1u);
  ASSERT_EQ(C.getColSpan(1), 1u);
  EXPECT_EQ(C[0][0], 1);
  EXPECT_EQ(C[1][0], 2);
  C.addRow(this->getDummyRow(2));
  EXPECT_EQ(C[2][0], 1);
  EXPECT_EQ(C[2][1], 2);
}

TYPED_TEST(MatrixTest, RowOps) {
  auto &M = this->SmallMatrix;
  auto &O = this->OtherSmall;
  JaggedArrayView<TypeParam> V{M};
  ASSERT_EQ(M.getNumRows(), 2u);
  ASSERT_EQ(V.getRowSpan(), 2u);
  V[0] = this->getDummyRow(2);
  V = JaggedArrayView<TypeParam>{M};
  EXPECT_EQ(V[0][0], 1);
  EXPECT_EQ(V[0][1], 2);
  ASSERT_EQ(M.getNumRows(), 2u);
  V.addRow({TypeParam(4), TypeParam(5)});
  ASSERT_EQ(M.getNumRows(), 3u);
  EXPECT_EQ(V[2][0], 4);
  EXPECT_EQ(V[2][1], 5);
  ASSERT_EQ(O.getNumRows(), 3u);
  JaggedArrayView<TypeParam> W{O};
  W.addRow(V[0]);
  ASSERT_EQ(O.getNumCols(), 3u);
  ASSERT_EQ(O.getNumRows(), 4u);
  W[3] = {TypeParam(7), TypeParam(8)};
  auto &WRow3 = W[3];
  WRow3 = WRow3.drop_back();
  ASSERT_EQ(WRow3[0], 7);
  WRow3 = WRow3.drop_back();
  EXPECT_TRUE(WRow3.empty());
  ASSERT_EQ(W.getColSpan(3), 0u);
  EXPECT_EQ(W[0][2], 0);
  EXPECT_EQ(W[1][2], 0);
  EXPECT_EQ(W[2][2], 0);
  W = JaggedArrayView<TypeParam>{O};
  EXPECT_EQ(W[0][2], 0);
  EXPECT_EQ(W[1][2], 0);
  EXPECT_EQ(W[2][2], 0);
  EXPECT_EQ(W[3][2], 0);
}

TYPED_TEST(MatrixTest, ResizeAssign) {
  auto &M = this->SmallMatrix;
  M.resize(3);
  ASSERT_EQ(M.getNumRows(), 3u);
  JaggedArrayView<TypeParam> V{M};
  V[0] = this->getDummyRow(2);
  V[1].copy_assign(V[0]);
  V[2].copy_assign(V[1]);
  ASSERT_EQ(M.getNumRows(), 3u);
  ASSERT_EQ(V.getRowSpan(), 3u);
  EXPECT_EQ(V[0], V[1]);
  EXPECT_EQ(V[2], V[1]);
  V = JaggedArrayView<TypeParam>{M};
  EXPECT_EQ(V[0], V[1]);
  EXPECT_EQ(V[2], V[1]);
}

TYPED_TEST(MatrixTest, ColSlice) {
  auto &M = this->OtherSmall;
  JaggedArrayView<TypeParam> V{M};
  V[0] = {TypeParam(3), TypeParam(7)};
  V[1] = {TypeParam(4), TypeParam(5)};
  V[2] = {TypeParam(8), TypeParam(9)};
  auto W = V.colSlice(1, 2);
  ASSERT_EQ(W.getRowSpan(), 3u);
  ASSERT_EQ(W.getColSpan(0), 1u);
  EXPECT_EQ(V[0][0], 3);
  EXPECT_EQ(V[0][1], 7);
  EXPECT_EQ(V[1][0], 4);
  EXPECT_EQ(V[1][1], 5);
  EXPECT_EQ(V[2][0], 8);
  EXPECT_EQ(V[2][1], 9);
  EXPECT_EQ(W[0][0], 7);
  EXPECT_EQ(W[1][0], 5);
  EXPECT_EQ(W[2][0], 9);
}

TYPED_TEST(MatrixTest, RowColSlice) {
  auto &M = this->OtherSmall;
  JaggedArrayView<TypeParam> V{M};
  V[0] = {TypeParam(3), TypeParam(7)};
  V[2] = {TypeParam(4), TypeParam(5)};
  auto W = V.rowSlice(0, 2).colSlice(0, 2);
  ASSERT_EQ(W.getRowSpan(), 2u);
  ASSERT_EQ(W.getColSpan(0), 2u);
  EXPECT_EQ(V[0][0], 3);
  EXPECT_EQ(V[0][1], 7);
  EXPECT_EQ(V[1][0], 0);
  EXPECT_EQ(V[1][1], 0);
  EXPECT_EQ(V[2][0], 4);
  EXPECT_EQ(V[2][1], 5);
  EXPECT_EQ(W[0][0], 3);
  EXPECT_EQ(W[0][1], 7);
  EXPECT_EQ(W[1][0], 0);
  EXPECT_EQ(W[1][1], 0);
}

TYPED_TEST(MatrixTest, NonWritingSwap) {
  auto &M = this->SmallMatrix;
  JaggedArrayView<TypeParam> V{M};
  V[0] = {TypeParam(3), TypeParam(7)};
  V[1] = {TypeParam(4), TypeParam(5)};
  V[0].swap(V[1]);
  EXPECT_EQ(V.lastRow()[0], 3);
  EXPECT_EQ(V.lastRow()[1], 7);
  EXPECT_EQ(V[0][0], 4);
  EXPECT_EQ(V[0][1], 5);
  EXPECT_EQ(V[1][0], 3);
  EXPECT_EQ(V[1][1], 7);
  auto W = JaggedArrayView<TypeParam>{M};
  EXPECT_EQ(W[0][0], 3);
  EXPECT_EQ(W[0][1], 7);
  EXPECT_EQ(W[1][0], 4);
  EXPECT_EQ(W[1][1], 5);
}

TYPED_TEST(MatrixTest, DropLastRow) {
  auto &M = this->OtherSmall;
  auto V = JaggedArrayView<TypeParam>{M};
  ASSERT_EQ(V.getRowSpan(), 3u);
  V.dropLastRow();
  ASSERT_EQ(V.getRowSpan(), 2u);
  V[0] = {TypeParam(19), TypeParam(7), TypeParam(3)};
  V[1] = {TypeParam(19), TypeParam(7), TypeParam(3)};
  ASSERT_EQ(V.getColSpan(1), 3u);
  V.addRow(V.lastRow());
  ASSERT_EQ(V.getRowSpan(), 3u);
  ASSERT_EQ(V.getColSpan(2), 3u);
  EXPECT_EQ(V[1], V.lastRow());
  V.dropLastRow();
  ASSERT_EQ(V.getRowSpan(), 2u);
  EXPECT_EQ(V[0], V.lastRow());
  V.addRow(this->getDummyRow(3));
  V.dropLastRow();
  V.addRow(this->getDummyRow(3));
  V.dropLastRow();
  ASSERT_EQ(V.getRowSpan(), 2u);
  EXPECT_EQ(V[0], V.lastRow());
  V.dropLastRow();
  V.dropLastRow();
  ASSERT_TRUE(V.empty());
  V.addRow({TypeParam(9), TypeParam(7), TypeParam(3)});
  V.addRow(this->getDummyRow(3));
  ASSERT_EQ(V.getRowSpan(), 2u);
  EXPECT_EQ(V.lastRow()[0], 1);
  EXPECT_EQ(V.lastRow()[1], 2);
  EXPECT_EQ(V.lastRow()[2], 3);
  EXPECT_EQ(V[0][0], 9);
  EXPECT_EQ(V[0][1], 7);
  EXPECT_EQ(V[0][2], 3);
  V.dropLastRow();
  V.addRow({TypeParam(21), TypeParam(22), TypeParam(23)});
  ASSERT_EQ(V.getRowSpan(), 2u);
  EXPECT_EQ(V.lastRow()[0], 21);
  EXPECT_EQ(V.lastRow()[1], 22);
  EXPECT_EQ(V.lastRow()[2], 23);
}

TYPED_TEST(MatrixTest, EraseLastRow) {
  auto &M = this->SmallMatrix;
  JaggedArrayView<TypeParam> V{M};
  V[0] = {TypeParam(3), TypeParam(7)};
  V[1] = {TypeParam(4), TypeParam(5)};
  V.eraseLastRow();
  ASSERT_EQ(M.size(), 2u);
  ASSERT_EQ(V.getRowSpan(), 1u);
  auto W = V.lastRow();
  ASSERT_EQ(W.size(), 2u);
  EXPECT_EQ(W[0], 3);
  EXPECT_EQ(W[1], 7);
  V.addRow({TypeParam(1), TypeParam(2)});
  ASSERT_EQ(V.getRowSpan(), 2u);
  V[0].copy_swap(V[1]);
  EXPECT_EQ(V[0][0], 1);
  EXPECT_EQ(V[0][1], 2);
  EXPECT_EQ(V.lastRow()[0], 3);
  EXPECT_EQ(V.lastRow()[1], 7);
  V.eraseLastRow();
  V.addRow({TypeParam(3), TypeParam(7)});
  ASSERT_EQ(V.getRowSpan(), 2u);
  EXPECT_EQ(V[0][0], 1);
  EXPECT_EQ(V[0][1], 2);
  EXPECT_EQ(V[1][0], 3);
  EXPECT_EQ(V[1][1], 7);
  V.eraseLastRow();
  ASSERT_EQ(V.getRowSpan(), 1u);
  EXPECT_EQ(V[0][0], 1);
  EXPECT_EQ(V[0][1], 2);
  V.eraseLastRow();
  EXPECT_TRUE(V.empty());
  EXPECT_TRUE(M.empty());
}

TYPED_TEST(MatrixTest, Iteration) {
  auto &M = this->SmallMatrix;
  M.resize(2);
  JaggedArrayView<TypeParam> V{M};
  for (const auto &[RowIdx, Row] : enumerate(V)) {
    V[RowIdx] = this->getDummyRow(2);
    for (const auto &[ColIdx, Col] : enumerate(Row)) {
      EXPECT_GT(V[RowIdx][ColIdx], 0);
    }
  }
}

TYPED_TEST(MatrixTest, VariableLengthColumns) {
  auto &M = this->ColumnInitMatrix;
  JaggedArrayView<TypeParam, 8, 16> V{M};
  ASSERT_EQ(V.empty(), true);
  size_t NumCols = 6;
  size_t NumRows = 3;
  SmallVector<TypeParam> ColumnVec;
  for (size_t Var = 0; Var < NumCols; ++Var)
    ColumnVec.emplace_back(Var + 1);
  ASSERT_EQ(ColumnVec.size(), NumCols);
  for (size_t RowIdx = 0; RowIdx < NumRows; ++RowIdx)
    V.addRow(ColumnVec);
  ASSERT_EQ(V.getMaxColSpan(), NumCols);
  V.addRow({TypeParam(19)});
  ASSERT_EQ(V.getColSpan(NumRows), 1u);
  V.dropLastRow();
  V.addRow(V.lastRow());
  ASSERT_EQ(V.getColSpan(NumRows), NumCols);
  EXPECT_EQ(V[NumRows - 1], V[NumRows]);
  V.dropLastRow();
  ASSERT_EQ(V.getRowSpan(), NumRows);
  ASSERT_EQ(M.getNumCols(), 8u);
  ASSERT_EQ(M.getNumRows(), NumRows + 2);
  V.dropLastRow();
  ASSERT_EQ(V.getRowSpan(), NumRows - 1);
  ASSERT_EQ(M.getNumRows(), NumRows + 2);
  ASSERT_EQ(V.getColSpan(1), NumCols);
  V[1][0] = 19;
  std::swap(V[0], V[1]);
  V[0].pop_back();
  V[1] = V[1].drop_back().drop_back();
  ASSERT_EQ(V.getColSpan(0), NumCols - 1);
  ASSERT_EQ(V.getColSpan(1), NumCols - 2);
  EXPECT_EQ(V[0][0], 19);
  EXPECT_EQ(V[1][0], 1);
  ASSERT_EQ(V.getMaxColSpan(), NumCols - 1);
  V = V.colSlice(0, 1).rowSlice(0, 2);
  ASSERT_EQ(V.getColSpan(0), 1u);
  ASSERT_EQ(V.getColSpan(1), 1u);
  EXPECT_EQ(V.getMaxColSpan(), 1u);
  ASSERT_EQ(V.getRowSpan(), 2u);
  EXPECT_EQ(V[0][0], 19);
  EXPECT_EQ(V[1][0], 1);
  V.dropLastRow();
  V.dropLastRow();
  EXPECT_TRUE(V.empty());
  EXPECT_EQ(M.size(), 40u);
}

TYPED_TEST(MatrixTest, LargeMatrixOps) {
  auto &M = this->LargeMatrix;
  ASSERT_EQ(M.getNumRows(), 16u);
  ASSERT_EQ(M.getNumCols(), 16u);
  JaggedArrayView<TypeParam, 16> V{M};
  V[0] = {TypeParam(1),  TypeParam(2), TypeParam(1),  TypeParam(4),
          TypeParam(5),  TypeParam(1), TypeParam(1),  TypeParam(8),
          TypeParam(9),  TypeParam(1), TypeParam(1),  TypeParam(12),
          TypeParam(13), TypeParam(1), TypeParam(15), TypeParam(1)};
  V[1] = this->getDummyRow(16);
  EXPECT_EQ(V[0][14], 15);
  EXPECT_EQ(V[0][3], 4);
  EXPECT_EQ(std::count(V[0].begin(), V[0].end(), 1), 8);
  EXPECT_TRUE(
      std::all_of(V[0].begin(), V[0].end(), [](auto &El) { return El > 0; }));
  auto W = V.rowSlice(0, 1).colSlice(2, 4);
  ASSERT_EQ(W.getRowSpan(), 1u);
  ASSERT_EQ(W.getColSpan(0), 2u);
  EXPECT_EQ(W[0][0], 1);
  EXPECT_EQ(W[0][1], 4);
}
