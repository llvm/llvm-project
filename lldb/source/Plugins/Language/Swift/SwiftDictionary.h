//===-- SwiftDictionary.h ---------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftDictionary_h_
#define liblldb_SwiftDictionary_h_

#include "lldb/lldb-forward.h"

#include "SwiftHashedContainer.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/DataFormatters/FormatClasses.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Target/Target.h"

#include "Plugins/Language/ObjC/NSDictionary.h"

namespace lldb_private {
namespace formatters {
namespace swift {

class DictionaryConfig: public HashedCollectionConfig {
public:
  static const DictionaryConfig &Get();

  static bool
  SummaryProvider(ValueObject &valobj, Stream &stream,
                  const TypeSummaryOptions &options);

  static SyntheticChildrenFrontEnd *
  SyntheticChildrenCreator(CXXSyntheticChildren *, lldb::ValueObjectSP);

private:
  DictionaryConfig();

protected:
  virtual CXXFunctionSummaryFormat::Callback
  GetSummaryProvider() const override {
    return DictionaryConfig::SummaryProvider;
  }
  
  virtual CXXSyntheticChildren::CreateFrontEndCallback
  GetSyntheticChildrenCreator() const override {
    return DictionaryConfig::SyntheticChildrenCreator;
  }

  virtual CXXSyntheticChildren::CreateFrontEndCallback
  GetCocoaSyntheticChildrenCreator() const override {
    return NSDictionarySyntheticFrontEndCreator;
  }
};    

}
}
}

#endif // liblldb_SwiftDictionary_h_
