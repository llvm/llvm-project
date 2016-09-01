//===-- MinidumpTypesTest.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Project includes
#include "Plugins/Process/minidump/MinidumpParser.h"
#include "Plugins/Process/minidump/MinidumpTypes.h"

// Other libraries and framework includes
#include "gtest/gtest.h"

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Host/FileSpec.h"

#include "llvm/ADT/Optional.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

// C includes

// C++ includes
#include <memory>

extern const char *TestMainArgv0;

using namespace lldb_private;
using namespace minidump;

class MinidumpParserTest : public testing::Test
{
public:
    void
    SetUp() override
    {
        llvm::StringRef dmp_folder = llvm::sys::path::parent_path(TestMainArgv0);
        inputs_folder = dmp_folder;
        llvm::sys::path::append(inputs_folder, "Inputs");
    }

    void
    SetUpData(const char *minidump_filename, size_t load_size = SIZE_MAX)
    {
        llvm::SmallString<128> filename = inputs_folder;
        llvm::sys::path::append(filename, minidump_filename);
        FileSpec minidump_file(filename.c_str(), false);
        lldb::DataBufferSP data_sp(minidump_file.MemoryMapFileContents(0, load_size));
        llvm::Optional<MinidumpParser> optional_parser = MinidumpParser::Create(data_sp);
        ASSERT_TRUE(optional_parser.hasValue());
        parser.reset(new MinidumpParser(optional_parser.getValue()));
        ASSERT_GT(parser->GetByteSize(), 0UL);
    }

    llvm::SmallString<128> inputs_folder;
    std::unique_ptr<MinidumpParser> parser;
};

TEST_F(MinidumpParserTest, GetThreads)
{
    SetUpData("linux-x86_64.dmp");
    llvm::Optional<std::vector<const MinidumpThread *>> thread_list;

    thread_list = parser->GetThreads();
    ASSERT_TRUE(thread_list.hasValue());
    ASSERT_EQ(1UL, thread_list->size());

    const MinidumpThread *thread = thread_list.getValue()[0];
    ASSERT_EQ(16001UL, thread->thread_id);
}

TEST_F(MinidumpParserTest, GetThreadsTruncatedFile)
{
    SetUpData("linux-x86_64.dmp", 200);
    llvm::Optional<std::vector<const MinidumpThread *>> thread_list;

    thread_list = parser->GetThreads();
    ASSERT_FALSE(thread_list.hasValue());
}

TEST_F(MinidumpParserTest, GetArchitecture)
{
    SetUpData("linux-x86_64.dmp");
    ASSERT_EQ(llvm::Triple::ArchType::x86_64, parser->GetArchitecture().GetTriple().getArch());
}

TEST_F(MinidumpParserTest, GetMiscInfo)
{
    SetUpData("linux-x86_64.dmp");
    const MinidumpMiscInfo *misc_info = parser->GetMiscInfo();
    ASSERT_EQ(nullptr, misc_info);
    // linux breakpad generated minidump files don't have misc info stream
}
