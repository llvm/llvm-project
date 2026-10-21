//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ExecutionContext.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <deque>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;

namespace {
class CallEdgeTest : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileMachO, SymbolFileDWARF,
                TypeSystemClang>
      subsystems;

public:
  void SetUp() override { m_module_sp = MakeModule(); }

protected:
  /// Build a module of its own from the shared test yaml. Each call yields a
  /// distinct Module, as two Targets would have.
  ModuleSP MakeModule() {
    auto file = TestFile::fromYamlFile("inlined-functions.yaml");
    if (!file) {
      ADD_FAILURE() << llvm::toString(file.takeError());
      return nullptr;
    }
    m_files.push_back(std::move(*file));
    return std::make_shared<Module>(m_files.back().moduleSpec());
  }

  std::deque<TestFile> m_files;
  ModuleSP m_module_sp;
};
} // namespace

TEST_F(CallEdgeTest, DirectCallEdgeCalleeDiesWithItsModule) {
  ModuleList images;
  images.Append(m_module_sp);

  DirectCallEdge edge("_Z4sum3iii", CallEdge::AddrType::AfterCall,
                      /*caller_address=*/0, /*is_tail_call=*/false, {});
  ExecutionContext exe_ctx;
  SymbolContext callee = edge.GetCallee(images, exe_ctx);
  ASSERT_TRUE(callee.function);
  EXPECT_EQ(callee.function->GetName(), ConstString("sum3(int, int, int)"));

  // Callee identity must remain stable while the module is loaded.
  EXPECT_EQ(edge.GetCallee(images, exe_ctx).function, callee.function);

  callee = SymbolContext();
  images.Clear();
  m_module_sp.reset();

  EXPECT_FALSE(edge.GetCallee(images, exe_ctx).function);
}

TEST_F(CallEdgeTest, CalleeKeepsItsModuleAlive) {
  ModuleList images;
  images.Append(m_module_sp);

  DirectCallEdge edge("_Z4sum3iii", CallEdge::AddrType::AfterCall,
                      /*caller_address=*/0, /*is_tail_call=*/false, {});
  ExecutionContext exe_ctx;
  SymbolContext callee = edge.GetCallee(images, exe_ctx);
  ASSERT_TRUE(callee.function);

  ModuleWP module_wp = m_module_sp;
  images.Clear();
  m_module_sp.reset();
  ASSERT_FALSE(module_wp.expired());

  EXPECT_EQ(callee.function->GetName(), ConstString("sum3(int, int, int)"));
  EXPECT_TRUE(callee.function->GetCallEdges().empty());
}

TEST_F(CallEdgeTest, UnresolvedSymbolYieldsNoCallee) {
  ModuleList images;
  images.Append(m_module_sp);

  DirectCallEdge edge("_Z7missingv", CallEdge::AddrType::AfterCall,
                      /*caller_address=*/0, /*is_tail_call=*/false, {});
  ExecutionContext exe_ctx;
  EXPECT_FALSE(edge.GetCallee(images, exe_ctx).function);
  EXPECT_FALSE(edge.GetCallee(images, exe_ctx).function);
}

TEST_F(CallEdgeTest, CalleeFollowsTheImageList) {
  ModuleSP other_module_sp = MakeModule();
  ASSERT_TRUE(other_module_sp);

  ModuleList images;
  images.Append(m_module_sp);
  ModuleList other_images;
  other_images.Append(other_module_sp);

  DirectCallEdge edge("_Z4sum3iii", CallEdge::AddrType::AfterCall,
                      /*caller_address=*/0, /*is_tail_call=*/false, {});
  ExecutionContext exe_ctx;
  SymbolContext callee = edge.GetCallee(images, exe_ctx);
  SymbolContext other_callee = edge.GetCallee(other_images, exe_ctx);
  ASSERT_TRUE(callee.function);
  ASSERT_TRUE(other_callee.function);

  EXPECT_EQ(callee.module_sp, m_module_sp);
  EXPECT_EQ(other_callee.module_sp, other_module_sp);
  EXPECT_NE(callee.function, other_callee.function);
}
