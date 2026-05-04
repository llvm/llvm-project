/*
 * TargetValue.h -- Access to target values using OMPD callbacks
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "omp-tools.h"
#include "ompd-private.h"
#include <stdlib.h>

#ifndef SRC_TARGET_VALUE_H_
#define SRC_TARGET_VALUE_H_

#ifdef __cplusplus

#include <cassert>
#include <map>
#include <string>

class TType;
class TValue;
class TBaseValue;

class TTypeFactory {
protected:
  std::map<ompd_address_space_context_t *, std::map<const char *, TType>>
      ttypes;

public:
  TTypeFactory() : ttypes() {}
  TType &getType(ompd_address_space_context_t *context, const char *typName,
                 ompd_addr_t segment = OMPD_SEGMENT_UNSPECIFIED);
};

static thread_local TTypeFactory tf = TTypeFactory();

class TType {
protected:
  ompd_size_t typeSize;
  std::map<const char *, ompd_size_t> fieldOffsets;
  std::map<const char *, ompd_size_t> fieldSizes;
  std::map<const char *, uint64_t> bitfieldMasks;
  ompd_addr_t descSegment;
  const char *typeName;
  ompd_address_space_context_t *context;
  bool isvoid;
  TType(ompd_address_space_context_t *context, const char *typeName,
        ompd_addr_t _segment = OMPD_SEGMENT_UNSPECIFIED);

public:
  TType(bool, ompd_addr_t _segment = OMPD_SEGMENT_UNSPECIFIED)
      : descSegment(_segment), isvoid(true) {}
  bool isVoid() const { return isvoid; }
  ompd_rc_t getElementOffset(const char *fieldName, ompd_size_t *offset);
  ompd_rc_t getElementSize(const char *fieldName, ompd_size_t *size);
  ompd_rc_t getBitfieldMask(const char *fieldName, uint64_t *bitfieldmask);
  ompd_rc_t getSize(ompd_size_t *size);
  friend TValue;
  friend TTypeFactory;
};

static TType nullType(true);

/**
 * class TError
 * As TValue is designed to concatenate operations, we use TError
 * to catch errors that might happen on each operation and provide
 * the according error code and which operation raised the error.
 */

class TError {
protected:
  ompd_rc_t errorCode;
  TError() : errorCode(ompd_rc_ok) {}
  TError(const ompd_rc_t &error) : errorCode(error) {}

public:
  std::string toString() {
    return std::string("TError messages not implemented yet");
  }
  friend TValue;
  friend TBaseValue;
};

/**
 * class TValue
 * This class encapsules the access to target values by using OMPD
 * callback functions. The member functions are designed to concatenate
 * the operations that are needed to access values from structures
 * e.g., _a[6]->_b._c would read like :
 * TValue(ctx,
 * "_a").cast("A",2).getArrayElement(6).access("_b").cast("B").access("_c")
 */

class TValue {
protected:
  TError errorState;
  TType *type;
  int pointerLevel;
  ompd_address_space_context_t *context;
  ompd_thread_context_t *tcontext;
  ompd_address_t symbolAddr;
  ompd_size_t fieldSize;

public:
  static const ompd_callbacks_t *callbacks;
  static ompd_device_type_sizes_t type_sizes;

  TValue() : errorState(ompd_rc_error) {}
  /**
   * Create a target value object from symbol name
   */
  TValue(ompd_address_space_context_t *_context, const char *_valueName,
         ompd_addr_t segment = OMPD_SEGMENT_UNSPECIFIED)
      : TValue(_context, NULL, _valueName, segment) {}

  TValue(ompd_address_space_context_t *context, ompd_thread_context_t *tcontext,
         const char *valueName, ompd_addr_t segment = OMPD_SEGMENT_UNSPECIFIED);
  /**
   * Create a target value object from target value address
   */
  TValue(ompd_address_space_context_t *_context, ompd_address_t _addr)
      : TValue(_context, NULL, _addr) {}
  TValue(ompd_address_space_context_t *context, ompd_thread_context_t *tcontext,
         ompd_address_t addr);
  /**
   * Cast the target value object to some type of typeName
   *
   * This call modifies the object and returns a reference to the modified
   * object
   */
  TValue &cast(const char *typeName);

  /**
   * Cast the target value object to some pointer of type typename
   * pointerlevel gives the number of *
   * e.g., char** would be: cast("char",2)
   *
   * This call modifies the object and returns a reference to the modified
   * object
   */
  TValue &cast(const char *typeName, int pointerLevel,
               ompd_addr_t segment = OMPD_SEGMENT_UNSPECIFIED);

  /**
   * Get the target address of the target value
   */
  ompd_rc_t getAddress(ompd_address_t *addr) const;
  /**
   * Get the raw memory copy of the target value
   */
  ompd_rc_t getRawValue(void *buf, int count);
  /**
   * Fetch a string copy from the target. "this" represents the pointer
   * that holds the value of a null terminated character string. "buf"
   * points to the destination string to be allocated and copied to.
   * Returns 'ompd_rc_error' to signify a truncated string or a target
   * read error.
   */
  ompd_rc_t getString(const char **buf);
  /**
   * Get a new target value object for the dereferenced target value
   * reduces the pointer level, uses the target value as new target address,
   * keeps the target type
   */
  TValue dereference() const;
  /**
   * Cast to a base type
   * Only values of base type may be read from target
   */
  TBaseValue castBase(ompd_target_prim_types_t baseType) const;
  /**
   * Cast to a base type
   * Get the size by fieldsize from runtime
   */
  TBaseValue castBase() const;
  /**
   * Cast to a base type
   * Get the size by name from the rtl
   */
  TBaseValue castBase(const char *varName);
  /**
   * Resolve field access for structs/unions
   * this supports both "->" and "." operator.
   */
  TValue access(const char *fieldName) const;
  /**
   * Tests for a field bit in a bitfield
   */
  ompd_rc_t check(const char *bitfieldName, ompd_word_t *isSet) const;
  /**
   * Get an array element
   */
  TValue getArrayElement(int elemNumber) const;
  /**
   * Get an element of a pointer array
   */
  TValue getPtrArrayElement(int elemNumber) const;
  /**
   * Did we raise some error yet?
   */
  bool gotError() const { return errorState.errorCode != ompd_rc_ok; }
  /**
   * Get the error code
   */
  ompd_rc_t getError() const { return errorState.errorCode; }
  /**
   * Did we raise some error yet?
   */
  std::string getErrorMessage() { return errorState.toString(); }
};

class TBaseValue : public TValue {
protected:
  ompd_size_t baseTypeSize = 0;
  TBaseValue(const TValue &, ompd_target_prim_types_t baseType);
  TBaseValue(const TValue &, ompd_size_t baseTypeSize);

public:
  ompd_rc_t getValue(void *buf, int count);
  template <typename T> ompd_rc_t getValue(T &buf);

  friend TValue;
};

template <typename T> ompd_rc_t TBaseValue::getValue(T &buf) {
  assert(sizeof(T) >= baseTypeSize);
  if (sizeof(T) == baseTypeSize)
    return getValue(&buf, 1);

  char tmp[sizeof(T)];
  ompd_rc_t ret = getValue(tmp, 1);
  switch (baseTypeSize) {
  case 1:
    buf = (T) * ((int8_t *)tmp);
    break;
  case 2:
    buf = (T) * ((int16_t *)tmp);
    break;
  case 4:
    buf = (T) * ((int32_t *)tmp);
    break;
  case 8:
    buf = (T) * ((int64_t *)tmp);
    break;
  default:
    assert(0 && "Invalid baseTypeSize");
    break;
  }
  return ret;
}

#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

#endif /*SRC_TARGET_VALUE_H_*/
