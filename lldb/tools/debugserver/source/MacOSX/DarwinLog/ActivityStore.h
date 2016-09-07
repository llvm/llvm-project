//===-- ActivityStore.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef ActivityStore_h
#define ActivityStore_h

#include <string>

#include "ActivityStreamSPI.h"

class ActivityStore {
public:
  virtual ~ActivityStore();

  virtual const char *GetActivityForID(os_activity_id_t activity_id) const = 0;

  virtual std::string
  GetActivityChainForID(os_activity_id_t activity_id) const = 0;

protected:
  ActivityStore();
};

#endif /* ActivityStore_h */
