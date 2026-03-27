//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/HostInfoWindows.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <windows.h>

// Find an unused drive letter to use as the subst target.
static wchar_t FindFreeDriveLetter() {
  DWORD in_use = ::GetLogicalDrives();
  for (wchar_t c = L'D'; c <= L'Z'; ++c)
    if (!(in_use & (1u << (c - L'A'))))
      return c;
  return L'\0';
}

class ResolveSubstDriveTest : public ::testing::Test {
protected:
  void SetUp() override {
    m_drive_letter = FindFreeDriveLetter();
    ASSERT_NE(m_drive_letter, L'\0') << "No free drive letter";
    m_drive[0] = m_drive_letter;
    m_drive[1] = L':';
    m_drive[2] = L'\0';
    ASSERT_TRUE(::DefineDosDeviceW(0, m_drive, L"C:\\SubstTestRoot"))
        << "DefineDosDeviceW failed: " << ::GetLastError();
  }

  void TearDown() override {
    if (m_drive_letter)
      ::DefineDosDeviceW(DDD_REMOVE_DEFINITION | DDD_EXACT_MATCH_ON_REMOVE,
                         m_drive, L"C:\\SubstTestRoot");
  }

  wchar_t m_drive_letter = L'\0';
  wchar_t m_drive[3] = {};
};

TEST_F(ResolveSubstDriveTest, ResolvesRealPath) {
  auto result = lldb_private::HostInfoWindows::ResolveSubstDrive(
      "C:\\SubstTestRoot\\foo\\a.out");
  ASSERT_TRUE(result.has_value());
  // drive_letter + ":\foo\a.out"
  // clang-format off
  std::string expected = {(char)m_drive_letter, ':', '\\', 'f','o','o','\\','a','.','o','u','t'};
  // clang-format on
  EXPECT_EQ(*result, expected);
}

TEST_F(ResolveSubstDriveTest, CaseInsensitive) {
  auto result = lldb_private::HostInfoWindows::ResolveSubstDrive(
      "c:\\substtestroot\\bar.dll");
  EXPECT_TRUE(result.has_value());
}

TEST_F(ResolveSubstDriveTest, NoMatchReturnsNullopt) {
  auto result = lldb_private::HostInfoWindows::ResolveSubstDrive(
      "C:\\UnrelatedDir\\foo.exe");
  EXPECT_FALSE(result.has_value());
}

TEST_F(ResolveSubstDriveTest, ExactRootMatch) {
  // Path equal to the subst target itself (no trailing component)
  auto result =
      lldb_private::HostInfoWindows::ResolveSubstDrive("C:\\SubstTestRoot");
  ASSERT_TRUE(result.has_value());
  std::string expected = {(char)m_drive_letter, ':'};
  EXPECT_EQ(*result, expected);
}

TEST_F(ResolveSubstDriveTest, PartialDirectoryNameNoFalseMatch) {
  // "C:\SubstTestRootExtra\..." must NOT match "C:\SubstTestRoot"
  auto result = lldb_private::HostInfoWindows::ResolveSubstDrive(
      "C:\\SubstTestRootExtra\\foo.exe");
  EXPECT_FALSE(result.has_value());
}