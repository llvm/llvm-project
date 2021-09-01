/*
 * TargetValue.cpp -- Access to target values using OMPD callbacks
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetValue.h"
#include "Debug.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

const ompd_callbacks_t *TValue::callbacks = NULL;
ompd_device_type_sizes_t TValue::type_sizes;

inline int ompd_sizeof(ompd_target_prim_types_t t) {
  assert(t != ompd_type_max && "ompd_type_max should not be used anywhere");
  assert(t != ompd_type_invalid && "request size of invalid type");

  return (((char *)&TValue::type_sizes)[(int)t]);
}

TType &TTypeFactory::getType(ompd_address_space_context_t *context,
                             const char *typeName, ompd_addr_t segment) {
  TType empty(true);

  if (ttypes.find(context) == ttypes.end()) {
    std::map<const char *, TType> empty;
    ttypes[context] = empty;
  }

  auto t = ttypes.find(context);
  auto i = t->second.find(typeName);
  if (i == t->second.end())
    i = t->second.insert(
        i, std::make_pair(typeName, TType(context, typeName, segment)));
  else
    i->second.context = context;

  return i->second;
}

TType::TType(ompd_address_space_context_t *_context, const char *_typeName,
             ompd_addr_t _segment)
    : typeSize(0), fieldOffsets(), descSegment(_segment), typeName(_typeName),
      context(_context), isvoid(false) {}

ompd_rc_t TType::getSize(ompd_size_t *size) {
  ompd_rc_t ret = ompd_rc_ok;
  if (typeSize == 0) {
    ompd_address_t symbolAddr;
    ompd_size_t tmpSize;
    std::stringstream ss;
    ss << "ompd_sizeof__" << typeName;

    ret = TValue::callbacks->symbol_addr_lookup(context, NULL, ss.str().c_str(),
                                                &symbolAddr, NULL);
    if (ret != ompd_rc_ok) {
      dout << "missing symbol " << ss.str()
           << " add this to ompd-specific.h:\nOMPD_SIZEOF(" << typeName
           << ") \\" << std::endl;
      return ret;
    }

    symbolAddr.segment = descSegment;

    ret = TValue::callbacks->read_memory(
        context, NULL, &symbolAddr, 1 * TValue::type_sizes.sizeof_long_long,
        &(tmpSize));
    if (ret != ompd_rc_ok)
      return ret;
    ret = TValue::callbacks->device_to_host(
        context, &tmpSize, TValue::type_sizes.sizeof_long_long, 1, &(typeSize));
  }
  *size = typeSize;
  return ret;
}

ompd_rc_t TType::getBitfieldMask(const char *fieldName,
                                 uint64_t *bitfieldmask) {
  ompd_rc_t ret = ompd_rc_ok;
  auto i = bitfieldMasks.find(fieldName);
  if (i == bitfieldMasks.end()) {
    uint64_t tmpMask, bitfieldMask;
    ompd_address_t symbolAddr;
    std::stringstream ss;
    ss << "ompd_bitfield__" << typeName << "__" << fieldName;
    ret = TValue::callbacks->symbol_addr_lookup(context, NULL, ss.str().c_str(),
                                                &symbolAddr, NULL);
    if (ret != ompd_rc_ok) {
      dout << "missing symbol " << ss.str()
           << " add this to ompd-specific.h:\nOMPD_BITFIELD(" << typeName << ","
           << fieldName << ") \\" << std::endl;
      return ret;
    }
    symbolAddr.segment = descSegment;

    ret = TValue::callbacks->read_memory(
        context, NULL, &symbolAddr, 1 * TValue::type_sizes.sizeof_long_long,
        &(tmpMask));
    if (ret != ompd_rc_ok)
      return ret;
    ret = TValue::callbacks->device_to_host(context, &(tmpMask),
                                            TValue::type_sizes.sizeof_long_long,
                                            1, &(bitfieldMask));
    if (ret != ompd_rc_ok) {
      return ret;
    }
    i = bitfieldMasks.insert(i, std::make_pair(fieldName, bitfieldMask));
  }
  *bitfieldmask = i->second;
  return ret;
}

ompd_rc_t TType::getElementOffset(const char *fieldName, ompd_size_t *offset) {
  ompd_rc_t ret = ompd_rc_ok;
  auto i = fieldOffsets.find(fieldName);
  if (i == fieldOffsets.end()) {
    ompd_size_t tmpOffset, fieldOffset;
    ompd_address_t symbolAddr;
    std::stringstream ss;
    ss << "ompd_access__" << typeName << "__" << fieldName;

    ret = TValue::callbacks->symbol_addr_lookup(context, NULL, ss.str().c_str(),
                                                &symbolAddr, NULL);
    if (ret != ompd_rc_ok) {
      dout << "missing symbol " << ss.str()
           << " add this to ompd-specific.h:\nOMPD_ACCESS(" << typeName << ","
           << fieldName << ") \\" << std::endl;
      return ret;
    }
    symbolAddr.segment = descSegment;

    ret = TValue::callbacks->read_memory(
        context, NULL, &symbolAddr, 1 * TValue::type_sizes.sizeof_long_long,
        &(tmpOffset));
    if (ret != ompd_rc_ok)
      return ret;
    ret = TValue::callbacks->device_to_host(context, &(tmpOffset),
                                            TValue::type_sizes.sizeof_long_long,
                                            1, &fieldOffset);
    if (ret != ompd_rc_ok) {
      return ret;
    }
    i = fieldOffsets.insert(i, std::make_pair(fieldName, fieldOffset));
  }
  *offset = i->second;
  return ret;
}

ompd_rc_t TType::getElementSize(const char *fieldName, ompd_size_t *size) {
  ompd_rc_t ret = ompd_rc_ok;
  auto i = fieldSizes.find(fieldName);
  if (i == fieldSizes.end()) {
    ompd_size_t tmpOffset, fieldSize;
    ompd_address_t symbolAddr;
    std::stringstream ss;
    ss << "ompd_sizeof__" << typeName << "__" << fieldName;

    ret = TValue::callbacks->symbol_addr_lookup(context, NULL, ss.str().c_str(),
                                                &symbolAddr, NULL);
    if (ret != ompd_rc_ok) {
      dout << "missing symbol " << ss.str()
           << " add this to ompd-specific.h:\nOMPD_ACCESS(" << typeName << ","
           << fieldName << ") \\" << std::endl;
      return ret;
    }
    symbolAddr.segment = descSegment;

    ret = TValue::callbacks->read_memory(
        context, NULL, &symbolAddr, 1 * TValue::type_sizes.sizeof_long_long,
        &(tmpOffset));
    if (ret != ompd_rc_ok)
      return ret;
    ret = TValue::callbacks->device_to_host(context, &tmpOffset,
                                            TValue::type_sizes.sizeof_long_long,
                                            1, &fieldSize);
    if (ret != ompd_rc_ok) {
      return ret;
    }
    i = fieldSizes.insert(i, std::make_pair(fieldName, fieldSize));
  }
  *size = i->second;
  return ret;
}

TValue::TValue(ompd_address_space_context_t *_context,
               ompd_thread_context_t *_tcontext, const char *_valueName,
               ompd_addr_t segment)
    : errorState(ompd_rc_ok), type(&nullType), pointerLevel(0),
      context(_context), tcontext(_tcontext), fieldSize(0) {
  errorState.errorCode = callbacks->symbol_addr_lookup(
      context, tcontext, _valueName, &symbolAddr, NULL);
  symbolAddr.segment = segment;
}

TValue::TValue(ompd_address_space_context_t *_context,
               ompd_thread_context_t *_tcontext, ompd_address_t addr)
    : errorState(ompd_rc_ok), type(&nullType), pointerLevel(0),
      context(_context), tcontext(_tcontext), symbolAddr(addr), fieldSize(0) {
  if (addr.address == 0)
    errorState.errorCode = ompd_rc_bad_input;
}

TValue &TValue::cast(const char *typeName) {
  if (gotError())
    return *this;
  type = &tf.getType(context, typeName, symbolAddr.segment);
  pointerLevel = 0;
  assert(!type->isVoid() && "cast to invalid type failed");
  return *this;
}

TValue &TValue::cast(const char *typeName, int _pointerLevel,
                     ompd_addr_t segment) {
  if (gotError())
    return *this;
  type = &tf.getType(context, typeName, symbolAddr.segment);
  pointerLevel = _pointerLevel;
  symbolAddr.segment = segment;
  assert(!type->isVoid() && "cast to invalid type failed");
  return *this;
}

TValue TValue::dereference() const {
  if (gotError())
    return *this;
  ompd_address_t tmpAddr;
  assert(!type->isVoid() && "cannot work with void");
  assert(pointerLevel > 0 && "cannot dereference non-pointer");
  TValue ret = *this;
  ret.pointerLevel--;
  ret.errorState.errorCode = callbacks->read_memory(
      context, tcontext, &symbolAddr, 1 * TValue::type_sizes.sizeof_pointer,
      &(tmpAddr.address));
  if (ret.errorState.errorCode != ompd_rc_ok)
    return ret;

  ret.errorState.errorCode = callbacks->device_to_host(
      context, &(tmpAddr.address), TValue::type_sizes.sizeof_pointer, 1,
      &(ret.symbolAddr.address));
  if (ret.errorState.errorCode != ompd_rc_ok) {
    return ret;
  }
  if (ret.symbolAddr.address == 0)
    ret.errorState.errorCode = ompd_rc_unsupported;
  return ret;
}

ompd_rc_t TValue::getAddress(ompd_address_t *addr) const {
  *addr = symbolAddr;
  if (symbolAddr.address == 0)
    return ompd_rc_unsupported;
  return errorState.errorCode;
}

ompd_rc_t TValue::getRawValue(void *buf, int count) {
  if (errorState.errorCode != ompd_rc_ok)
    return errorState.errorCode;
  ompd_size_t size;
  errorState.errorCode = type->getSize(&size);
  if (errorState.errorCode != ompd_rc_ok)
    return errorState.errorCode;

  errorState.errorCode =
      callbacks->read_memory(context, tcontext, &symbolAddr, size, buf);
  return errorState.errorCode;
}

ompd_rc_t TValue::getString(const char **buf) {
  *buf = 0;
  if (gotError())
    return getError();

  TValue strValue = dereference();
  if (strValue.gotError()) {
    return strValue.getError();
  }

  if (!callbacks) {
    return ompd_rc_error;
  }
  ompd_rc_t ret;
#define BUF_LEN 512
  char *string_buffer;

  // Allocate an extra byte, but pass only BUF_LEN to the tool
  // so that we can detect truncation later.
  ret = callbacks->alloc_memory(BUF_LEN + 1, (void **)&string_buffer);
  if (ret != ompd_rc_ok) {
    return ret;
  }
  string_buffer[BUF_LEN] = '\0';

  // TODO: if we have not read in the complete string, we need to realloc
  // 'string_buffer' and attempt reading again repeatedly till the entire string
  // is read in.
  ret = callbacks->read_string(context, tcontext, &strValue.symbolAddr, BUF_LEN,
                               (void *)string_buffer);
  *buf = string_buffer;
  // Check for truncation. The standard specifies that if a null byte is not
  // among the first 'nbytes' bytes, the string placed in the buffer is not
  // null-terminated. 'nbytes' is BUF_LEN in this case.
  if (ret == ompd_rc_ok && strlen(string_buffer) == BUF_LEN) {
    return ompd_rc_error;
  }
  return ret;
}

TBaseValue TValue::castBase(const char *varName) {
  ompd_size_t size;
  errorState.errorCode =
      tf.getType(context, varName, symbolAddr.segment).getSize(&size);
  return TBaseValue(*this, size);
}

TBaseValue TValue::castBase() const {
  if (pointerLevel > 0)
    return TBaseValue(*this, type_sizes.sizeof_pointer);
  return TBaseValue(*this, fieldSize);
}

TBaseValue TValue::castBase(ompd_target_prim_types_t baseType) const {
  return TBaseValue(*this, baseType);
}

TValue TValue::access(const char *fieldName) const {
  if (gotError())
    return *this;
  TValue ret = *this;
  assert(pointerLevel < 2 && "access to field element of pointer array failed");
  if (pointerLevel == 1) // -> operator
    ret = ret.dereference();
  // we use *this for . operator
  ompd_size_t offset;
  ret.errorState.errorCode = type->getElementOffset(fieldName, &offset);
  ret.errorState.errorCode = type->getElementSize(fieldName, &(ret.fieldSize));
  ret.symbolAddr.address += offset;

  return ret;
}

ompd_rc_t TValue::check(const char *bitfieldName, ompd_word_t *isSet) const {
  if (gotError())
    return getError();
  int bitfield;
  uint64_t bitfieldmask;
  ompd_rc_t ret = this->castBase(ompd_type_int).getValue(&bitfield, 1);
  if (ret != ompd_rc_ok)
    return ret;
  ret = type->getBitfieldMask(bitfieldName, &bitfieldmask);
  *isSet = ((bitfield & bitfieldmask) != 0);
  return ret;
}

TValue TValue::getArrayElement(int elemNumber) const {
  if (gotError())
    return *this;
  TValue ret;
  if (pointerLevel > 0) {
    ret = dereference();
  } else {
    ret = *this;
  }
  if (ret.pointerLevel == 0) {
    ompd_size_t size;
    ret.errorState.errorCode = type->getSize(&size);
    ret.symbolAddr.address += elemNumber * size;
  } else {
    ret.symbolAddr.address += elemNumber * type_sizes.sizeof_pointer;
  }
  return ret;
}

TValue TValue::getPtrArrayElement(int elemNumber) const {
  if (gotError()) {
    return *this;
  }
  assert(pointerLevel > 0 && "This only works on arrays of pointers");
  TValue ret = *this;
  ret.symbolAddr.address += elemNumber * type_sizes.sizeof_pointer;
  return ret;
}

TBaseValue::TBaseValue(const TValue &_tvalue,
                       ompd_target_prim_types_t _baseType)
    : TValue(_tvalue), baseTypeSize(ompd_sizeof(_baseType)) {}
TBaseValue::TBaseValue(const TValue &_tvalue, ompd_size_t _baseTypeSize)
    : TValue(_tvalue), baseTypeSize(_baseTypeSize) {}

ompd_rc_t TBaseValue::getValue(void *buf, int count) {
  if (errorState.errorCode != ompd_rc_ok)
    return errorState.errorCode;
  errorState.errorCode = callbacks->read_memory(context, tcontext, &symbolAddr,
                                                count * baseTypeSize, buf);
  if (errorState.errorCode != ompd_rc_ok)
    return errorState.errorCode;
  errorState.errorCode =
      callbacks->device_to_host(context, buf, baseTypeSize, count, buf);
  return errorState.errorCode;
}
