//===---------------- PBQP.cpp --------- PBQP Solver ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Developed by:                   Bernhard Scholz
//                             The University of Sydney
//                         http://www.it.usyd.edu.au/~scholz
//===----------------------------------------------------------------------===//

// TODO:
//
//  * Default to null costs on vector initialisation?
//  * C++-ify the rest of the solver.

#ifndef LLVM_CODEGEN_PBQPSOLVER_H
#define LLVM_CODEGEN_PBQPSOLVER_H

#include <cassert>
#include <algorithm>
#include <functional>

namespace llvm {

//! \brief Floating point type to use in PBQP solver.
typedef double PBQPNum;

//! \brief PBQP Vector class.
class PBQPVector {
public:

  //! \brief Construct a PBQP vector of the given size.
  explicit PBQPVector(unsigned length) :
    length(length), data(new PBQPNum[length]) {
    std::fill(data, data + length, 0);
  }

  //! \brief Copy construct a PBQP vector.
  PBQPVector(const PBQPVector &v) :
    length(v.length), data(new PBQPNum[length]) {
    std::copy(v.data, v.data + length, data);
  }

  ~PBQPVector() { delete[] data; }

  //! \brief Assignment operator.
  PBQPVector& operator=(const PBQPVector &v) {
    delete[] data;
    length = v.length;
    data = new PBQPNum[length];
    std::copy(v.data, v.data + length, data);
    return *this;
  }

  //! \brief Return the length of the vector
  unsigned getLength() const throw () {
    return length;
  }

  //! \brief Element access.
  PBQPNum& operator[](unsigned index) {
    assert(index < length && "PBQPVector element access out of bounds.");
    return data[index];
  }

  //! \brief Const element access.
  const PBQPNum& operator[](unsigned index) const {
    assert(index < length && "PBQPVector element access out of bounds.");
    return data[index];
  }

  //! \brief Add another vector to this one.
  PBQPVector& operator+=(const PBQPVector &v) {
    assert(length == v.length && "PBQPVector length mismatch.");
    std::transform(data, data + length, v.data, data, std::plus<PBQPNum>()); 
    return *this;
  }

  //! \brief Subtract another vector from this one.
  PBQPVector& operator-=(const PBQPVector &v) {
    assert(length == v.length && "PBQPVector length mismatch.");
    std::transform(data, data + length, v.data, data, std::minus<PBQPNum>()); 
    return *this;
  }

  //! \brief Returns the index of the minimum value in this vector
  unsigned minIndex() const {
    return std::min_element(data, data + length) - data;
  }

private:
  unsigned length;
  PBQPNum *data;
};


//! \brief PBQP Matrix class
class PBQPMatrix {
public:

  //! \brief Construct a PBQP Matrix with the given dimensions.
  PBQPMatrix(unsigned rows, unsigned cols) :
    rows(rows), cols(cols), data(new PBQPNum[rows * cols]) {
    std::fill(data, data + (rows * cols), 0);
  }

  //! \brief Copy construct a PBQP matrix.
  PBQPMatrix(const PBQPMatrix &m) :
    rows(m.rows), cols(m.cols), data(new PBQPNum[rows * cols]) {
    std::copy(m.data, m.data + (rows * cols), data);  
  }

  ~PBQPMatrix() { delete[] data; }

  //! \brief Assignment operator.
  PBQPMatrix& operator=(const PBQPMatrix &m) {
    delete[] data;
    rows = m.rows; cols = m.cols;
    data = new PBQPNum[rows * cols];
    std::copy(m.data, m.data + (rows * cols), data);
    return *this;
  }

  //! \brief Return the number of rows in this matrix.
  unsigned getRows() const throw () { return rows; }

  //! \brief Return the number of cols in this matrix.
  unsigned getCols() const throw () { return cols; }

  //! \brief Matrix element access.
  PBQPNum* operator[](unsigned r) {
    assert(r < rows && "Row out of bounds.");
    return data + (r * cols);
  }

  //! \brief Matrix element access.
  const PBQPNum* operator[](unsigned r) const {
    assert(r < rows && "Row out of bounds.");
    return data + (r * cols);
  }

  //! \brief Returns the given row as a vector.
  PBQPVector getRowAsVector(unsigned r) const {
    PBQPVector v(cols);
    for (unsigned c = 0; c < cols; ++c)
      v[c] = (*this)[r][c];
    return v; 
  }

  //! \brief Reset the matrix to the given value.
  PBQPMatrix& reset(PBQPNum val = 0) {
    std::fill(data, data + (rows * cols), val);
    return *this;
  }

  //! \brief Set a single row of this matrix to the given value.
  PBQPMatrix& setRow(unsigned r, PBQPNum val) {
    assert(r < rows && "Row out of bounds.");
    std::fill(data + (r * cols), data + ((r + 1) * cols), val);
    return *this;
  }

  //! \brief Set a single column of this matrix to the given value.
  PBQPMatrix& setCol(unsigned c, PBQPNum val) {
    assert(c < cols && "Column out of bounds.");
    for (unsigned r = 0; r < rows; ++r)
      (*this)[r][c] = val;
    return *this;
  }

  //! \brief Matrix transpose.
  PBQPMatrix transpose() const {
    PBQPMatrix m(cols, rows);
    for (unsigned r = 0; r < rows; ++r)
      for (unsigned c = 0; c < cols; ++c)
        m[c][r] = (*this)[r][c];
    return m;
  }

  //! \brief Returns the diagonal of the matrix as a vector.
  //!
  //! Matrix must be square.
  PBQPVector diagonalize() const {
    assert(rows == cols && "Attempt to diagonalize non-square matrix.");

    PBQPVector v(rows);
    for (unsigned r = 0; r < rows; ++r)
      v[r] = (*this)[r][r];
    return v;
  } 

  //! \brief Add the given matrix to this one.
  PBQPMatrix& operator+=(const PBQPMatrix &m) {
    assert(rows == m.rows && cols == m.cols &&
           "Matrix dimensions mismatch.");
    std::transform(data, data + (rows * cols), m.data, data,
                   std::plus<PBQPNum>());
    return *this;
  }

  //! \brief Returns the minimum of the given row
  PBQPNum getRowMin(unsigned r) const {
    assert(r < rows && "Row out of bounds");
    return *std::min_element(data + (r * cols), data + ((r + 1) * cols));
  }

  //! \brief Returns the minimum of the given column
  PBQPNum getColMin(unsigned c) const {
    PBQPNum minElem = (*this)[0][c];
    for (unsigned r = 1; r < rows; ++r)
      if ((*this)[r][c] < minElem) minElem = (*this)[r][c];
    return minElem;
  }

  //! \brief Subtracts the given scalar from the elements of the given row.
  PBQPMatrix& subFromRow(unsigned r, PBQPNum val) {
    assert(r < rows && "Row out of bounds");
    std::transform(data + (r * cols), data + ((r + 1) * cols),
                   data + (r * cols),
                   std::bind2nd(std::minus<PBQPNum>(), val));
    return *this;
  }

  //! \brief Subtracts the given scalar from the elements of the given column.
  PBQPMatrix& subFromCol(unsigned c, PBQPNum val) {
    for (unsigned r = 0; r < rows; ++r)
      (*this)[r][c] -= val;
    return *this;
  }

  //! \brief Returns true if this is a zero matrix.
  bool isZero() const {
    return find_if(data, data + (rows * cols),
                   std::bind2nd(std::not_equal_to<PBQPNum>(), 0)) ==
                     data + (rows * cols);
  }

private:
  unsigned rows, cols;
  PBQPNum *data;
};

#define EPS (1E-8)

#ifndef PBQP_TYPE
#define PBQP_TYPE
struct pbqp;
typedef struct pbqp pbqp;
#endif

/*****************
 * PBQP routines *
 *****************/

/* allocate pbqp problem */
pbqp *alloc_pbqp(int num);

/* add node costs */
void add_pbqp_nodecosts(pbqp *this_,int u, PBQPVector *costs);

/* add edge mat */
void add_pbqp_edgecosts(pbqp *this_,int u,int v,PBQPMatrix *costs);

/* solve PBQP problem */
void solve_pbqp(pbqp *this_);

/* get solution of a node */
int get_pbqp_solution(pbqp *this_,int u);

/* alloc PBQP */
pbqp *alloc_pbqp(int num);

/* free PBQP */
void free_pbqp(pbqp *this_);

/* is optimal */
bool is_pbqp_optimal(pbqp *this_);

}
#endif
