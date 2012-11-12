//===------ Math.h - PBQP Vector and Matrix classes -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PBQP_MATH_H 
#define LLVM_CODEGEN_PBQP_MATH_H

#include <cassert>
#include <algorithm>
#include <functional>

namespace PBQP {

typedef float PBQPNum;

/// \brief PBQP Vector class.
class Vector {
  public:

    /// \brief Construct a PBQP vector of the given size.
    explicit Vector(unsigned length) :
      length(length), data(new PBQPNum[length]) {
      }

    /// \brief Construct a PBQP vector with initializer.
    Vector(unsigned length, PBQPNum initVal) :
      length(length), data(new PBQPNum[length]) {
        std::fill(data, data + length, initVal);
      }

    /// \brief Copy construct a PBQP vector.
    Vector(const Vector &v) :
      length(v.length), data(new PBQPNum[length]) {
        std::copy(v.data, v.data + length, data);
      }

    /// \brief Destroy this vector, return its memory.
    ~Vector() { delete[] data; }

    /// \brief Assignment operator.
    Vector& operator=(const Vector &v) {
      delete[] data;
      length = v.length;
      data = new PBQPNum[length];
      std::copy(v.data, v.data + length, data);
      return *this;
    }

    /// \brief Return the length of the vector
    unsigned getLength() const {
      return length;
    }

    /// \brief Element access.
    PBQPNum& operator[](unsigned index) {
      assert(index < length && "Vector element access out of bounds.");
      return data[index];
    }

    /// \brief Const element access.
    const PBQPNum& operator[](unsigned index) const {
      assert(index < length && "Vector element access out of bounds.");
      return data[index];
    }

    /// \brief Add another vector to this one.
    Vector& operator+=(const Vector &v) {
      assert(length == v.length && "Vector length mismatch.");
      std::transform(data, data + length, v.data, data, std::plus<PBQPNum>()); 
      return *this;
    }

    /// \brief Subtract another vector from this one.
    Vector& operator-=(const Vector &v) {
      assert(length == v.length && "Vector length mismatch.");
      std::transform(data, data + length, v.data, data, std::minus<PBQPNum>()); 
      return *this;
    }

    /// \brief Returns the index of the minimum value in this vector
    unsigned minIndex() const {
      return std::min_element(data, data + length) - data;
    }

  private:
    unsigned length;
    PBQPNum *data;
};

/// \brief Output a textual representation of the given vector on the given
///        output stream.
template <typename OStream>
OStream& operator<<(OStream &os, const Vector &v) {
  assert((v.getLength() != 0) && "Zero-length vector badness.");

  os << "[ " << v[0];
  for (unsigned i = 1; i < v.getLength(); ++i) {
    os << ", " << v[i];
  }
  os << " ]";

  return os;
} 


/// \brief PBQP Matrix class
class Matrix {
  public:

    /// \brief Construct a PBQP Matrix with the given dimensions.
    Matrix(unsigned rows, unsigned cols) :
      rows(rows), cols(cols), data(new PBQPNum[rows * cols]) {
    }

    /// \brief Construct a PBQP Matrix with the given dimensions and initial
    /// value.
    Matrix(unsigned rows, unsigned cols, PBQPNum initVal) :
      rows(rows), cols(cols), data(new PBQPNum[rows * cols]) {
        std::fill(data, data + (rows * cols), initVal);
    }

    /// \brief Copy construct a PBQP matrix.
    Matrix(const Matrix &m) :
      rows(m.rows), cols(m.cols), data(new PBQPNum[rows * cols]) {
        std::copy(m.data, m.data + (rows * cols), data);  
    }

    /// \brief Destroy this matrix, return its memory.
    ~Matrix() { delete[] data; }

    /// \brief Assignment operator.
    Matrix& operator=(const Matrix &m) {
      delete[] data;
      rows = m.rows; cols = m.cols;
      data = new PBQPNum[rows * cols];
      std::copy(m.data, m.data + (rows * cols), data);
      return *this;
    }

    /// \brief Return the number of rows in this matrix.
    unsigned getRows() const { return rows; }

    /// \brief Return the number of cols in this matrix.
    unsigned getCols() const { return cols; }

    /// \brief Matrix element access.
    PBQPNum* operator[](unsigned r) {
      assert(r < rows && "Row out of bounds.");
      return data + (r * cols);
    }

    /// \brief Matrix element access.
    const PBQPNum* operator[](unsigned r) const {
      assert(r < rows && "Row out of bounds.");
      return data + (r * cols);
    }

    /// \brief Returns the given row as a vector.
    Vector getRowAsVector(unsigned r) const {
      Vector v(cols);
      for (unsigned c = 0; c < cols; ++c)
        v[c] = (*this)[r][c];
      return v; 
    }

    /// \brief Returns the given column as a vector.
    Vector getColAsVector(unsigned c) const {
      Vector v(rows);
      for (unsigned r = 0; r < rows; ++r)
        v[r] = (*this)[r][c];
      return v;
    }

    /// \brief Reset the matrix to the given value.
    Matrix& reset(PBQPNum val = 0) {
      std::fill(data, data + (rows * cols), val);
      return *this;
    }

    /// \brief Set a single row of this matrix to the given value.
    Matrix& setRow(unsigned r, PBQPNum val) {
      assert(r < rows && "Row out of bounds.");
      std::fill(data + (r * cols), data + ((r + 1) * cols), val);
      return *this;
    }

    /// \brief Set a single column of this matrix to the given value.
    Matrix& setCol(unsigned c, PBQPNum val) {
      assert(c < cols && "Column out of bounds.");
      for (unsigned r = 0; r < rows; ++r)
        (*this)[r][c] = val;
      return *this;
    }

    /// \brief Matrix transpose.
    Matrix transpose() const {
      Matrix m(cols, rows);
      for (unsigned r = 0; r < rows; ++r)
        for (unsigned c = 0; c < cols; ++c)
          m[c][r] = (*this)[r][c];
      return m;
    }

    /// \brief Returns the diagonal of the matrix as a vector.
    ///
    /// Matrix must be square.
    Vector diagonalize() const {
      assert(rows == cols && "Attempt to diagonalize non-square matrix.");

      Vector v(rows);
      for (unsigned r = 0; r < rows; ++r)
        v[r] = (*this)[r][r];
      return v;
    } 

    /// \brief Add the given matrix to this one.
    Matrix& operator+=(const Matrix &m) {
      assert(rows == m.rows && cols == m.cols &&
          "Matrix dimensions mismatch.");
      std::transform(data, data + (rows * cols), m.data, data,
          std::plus<PBQPNum>());
      return *this;
    }

    /// \brief Returns the minimum of the given row
    PBQPNum getRowMin(unsigned r) const {
      assert(r < rows && "Row out of bounds");
      return *std::min_element(data + (r * cols), data + ((r + 1) * cols));
    }

    /// \brief Returns the minimum of the given column
    PBQPNum getColMin(unsigned c) const {
      PBQPNum minElem = (*this)[0][c];
      for (unsigned r = 1; r < rows; ++r)
        if ((*this)[r][c] < minElem) minElem = (*this)[r][c];
      return minElem;
    }

    /// \brief Subtracts the given scalar from the elements of the given row.
    Matrix& subFromRow(unsigned r, PBQPNum val) {
      assert(r < rows && "Row out of bounds");
      std::transform(data + (r * cols), data + ((r + 1) * cols),
          data + (r * cols),
          std::bind2nd(std::minus<PBQPNum>(), val));
      return *this;
    }

    /// \brief Subtracts the given scalar from the elements of the given column.
    Matrix& subFromCol(unsigned c, PBQPNum val) {
      for (unsigned r = 0; r < rows; ++r)
        (*this)[r][c] -= val;
      return *this;
    }

    /// \brief Returns true if this is a zero matrix.
    bool isZero() const {
      return find_if(data, data + (rows * cols),
          std::bind2nd(std::not_equal_to<PBQPNum>(), 0)) ==
        data + (rows * cols);
    }

  private:
    unsigned rows, cols;
    PBQPNum *data;
};

/// \brief Output a textual representation of the given matrix on the given
///        output stream.
template <typename OStream>
OStream& operator<<(OStream &os, const Matrix &m) {

  assert((m.getRows() != 0) && "Zero-row matrix badness.");

  for (unsigned i = 0; i < m.getRows(); ++i) {
    os << m.getRowAsVector(i);
  }

  return os;
}

}

#endif // LLVM_CODEGEN_PBQP_MATH_H
