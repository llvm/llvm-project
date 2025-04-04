//===- StaticMemoryPlanning.cpp - Tests for StaticMemoryPlanning-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/StaticMemoryPlanning.h"

#include "gtest/gtest.h"
#include <iostream>

using namespace mlir;
TEST(static_memory_planner, TestStaticMemoryPlanning) {
  /*
  {0}                   {160}               {280}                 {450}
  |           0         |           1        |          2          |
  |    3    |   4/5     |                                          |
  |    7    |                                     6                |
            {100}
  */
  memoryplan::Traces traces = {{0, 100}, {1, 120}, {2, 100}, {0, 0},
                               {3, 50},  {4, 60},  {2, 0},   {4, 0},
                               {8, 100}, {8, 0},   {5, 60},  {5, 0},
                               {1, 0},   {6, 350}, {3, 0},   {7, 100}};
  std::unordered_map<uintptr_t, size_t> out;
  std::unordered_map<uintptr_t, std::vector<uintptr_t>> inplace_selection;
  size_t total = memoryplan::scheduleMemoryAllocations(
      traces, 1, false, memoryplan::InplaceInfoMap(), out, inplace_selection);
  std::unordered_map<uintptr_t, size_t> expected_out = {
      {0, 0},   {1, 160}, {2, 280}, {3, 0},  {4, 100},
      {5, 100}, {6, 100}, {7, 0},   {8, 280}};
  EXPECT_EQ(total, 450UL);
  EXPECT_EQ(out, expected_out);

  total = memoryplan::scheduleMemoryAllocations(
      traces, 1, true, memoryplan::InplaceInfoMap(), out, inplace_selection);
  expected_out = {{0, 0},   {1, 160}, {2, 280}, {3, 0},  {4, 100},
                  {5, 100}, {6, 100}, {7, 0},   {8, 280}};
  EXPECT_EQ(total, 450UL);
  EXPECT_EQ(out, expected_out);
}

TEST(static_memory_planner, TestStaticMemoryPlanningInplace) {
  using namespace memoryplan;
  using inplace_outdata = std::unordered_map<uintptr_t, std::vector<uintptr_t>>;
  using inplace_data = InplaceInfoMap;
  // simple inplace (need merge + split)
  {
    memoryplan::Traces traces = {{1, 120}, {2, 100}, {3, 200}, {1, 0},
                                 {2, 0},   {3, 0},   {4, 220}, {4, 0}};
    std::unordered_map<uintptr_t, size_t> out;
    inplace_outdata inplace_selection;
    inplace_data inplace_hint = {
        {3, {{1, InplaceKind::FREE}, {2, InplaceKind::FREE}}}};
    size_t total = memoryplan::scheduleMemoryAllocations(
        traces, 1, false, inplace_hint, out, inplace_selection);
    std::unordered_map<uintptr_t, size_t> expected_out = {
        {1, 0}, {2, 120}, {3, 0}, {4, 0}};
    EXPECT_EQ(total, 220UL);
    EXPECT_EQ(out, expected_out);

    inplace_outdata expected_inplace = {{3, {1, 2}}};
    EXPECT_EQ(inplace_selection, expected_inplace);
  }

  // inplace extend
  {
    memoryplan::Traces traces = {{1, 120}, {2, 100}, {4, 250}, {3, 250},
                                 {1, 0},   {2, 0},   {3, 0},   {4, 0}};
    std::unordered_map<uintptr_t, size_t> out;
    inplace_outdata inplace_selection;
    inplace_data inplace_hint = {
        {3, {{1, InplaceKind::FREE}, {2, InplaceKind::FREE}}}};
    size_t total = memoryplan::scheduleMemoryAllocations(
        traces, 1, false, inplace_hint, out, inplace_selection);
    std::unordered_map<uintptr_t, size_t> expected_out = {
        {1, 0}, {2, 120}, {3, 0}, {4, 250}};
    EXPECT_EQ(total, 500UL);
    EXPECT_EQ(out, expected_out);

    inplace_outdata expected_inplace = {{3, {1, 2}}};
    EXPECT_EQ(inplace_selection, expected_inplace);
  }

  // inplace 2 buffers into one
  {
    memoryplan::Traces traces = {{1, 120}, {2, 100}, {3, 150}, {4, 50}, {5, 10},
                                 {1, 0},   {2, 0},   {3, 0},   {4, 0},  {5, 0}};
    std::unordered_map<uintptr_t, size_t> out;
    inplace_outdata inplace_selection;
    inplace_data inplace_hint = {
        {3, {{1, InplaceKind::FREE}, {2, InplaceKind::FREE}}},
        {4, {{1, InplaceKind::FREE}, {2, InplaceKind::FREE}}}};
    size_t total = memoryplan::scheduleMemoryAllocations(
        traces, 1, false, inplace_hint, out, inplace_selection);
    std::unordered_map<uintptr_t, size_t> expected_out = {
        {1, 0}, {2, 120}, {3, 0}, {4, 150}, {5, 220}};
    EXPECT_EQ(total, 230UL);
    EXPECT_EQ(out, expected_out);

    inplace_outdata expected_inplace = {{3, {1, 2}}, {4, {2}}};
    EXPECT_EQ(inplace_selection, expected_inplace);
  }

  // inplace 2 buffers into one, but require zero offset
  {
    memoryplan::Traces traces = {{1, 120}, {2, 100}, {3, 150}, {4, 50}, {5, 10},
                                 {1, 0},   {2, 0},   {3, 0},   {4, 0},  {5, 0}};
    std::unordered_map<uintptr_t, size_t> out;
    inplace_outdata inplace_selection;
    inplace_data inplace_hint = {
        {3, {{1, InplaceKind::FREE}, {2, InplaceKind::ZERO_OFFSET}}},
        {4, {{1, InplaceKind::FREE}, {2, InplaceKind::FREE}}}};
    size_t total = memoryplan::scheduleMemoryAllocations(
        traces, 1, false, inplace_hint, out, inplace_selection);
    std::unordered_map<uintptr_t, size_t> expected_out = {
        {1, 0}, {2, 150}, {3, 0}, {4, 150}, {5, 250}};
    EXPECT_EQ(total, 260UL);
    EXPECT_EQ(out, expected_out);

    inplace_outdata expected_inplace = {{3, {1}}, {4, {2}}};
    EXPECT_EQ(inplace_selection, expected_inplace);
  }

  // inplace 2 buffers into one, but require zero offset for split buffer
  // buffer4 cannot reuse buffer 2 because it requires zero offset
  {
    memoryplan::Traces traces = {{1, 120}, {2, 100}, {3, 150}, {4, 50}, {5, 10},
                                 {1, 0},   {2, 0},   {3, 0},   {4, 0},  {5, 0}};
    std::unordered_map<uintptr_t, size_t> out;
    inplace_outdata inplace_selection;
    inplace_data inplace_hint = {
        {3, {{1, InplaceKind::FREE}, {2, InplaceKind::FREE}}},
        {4, {{1, InplaceKind::FREE}, {2, InplaceKind::ZERO_OFFSET}}}};
    size_t total = memoryplan::scheduleMemoryAllocations(
        traces, 1, false, inplace_hint, out, inplace_selection);
    std::unordered_map<uintptr_t, size_t> expected_out = {
        {1, 0}, {2, 120}, {3, 0}, {4, 220}, {5, 270}};
    EXPECT_EQ(total, 280UL);
    EXPECT_EQ(out, expected_out);

    inplace_outdata expected_inplace = {{3, {1, 2}}};
    EXPECT_EQ(inplace_selection, expected_inplace);
  }

  // merge free to the right
  {
    memoryplan::Traces traces = {{1, 120}, {2, 100}, {3, 150}, {2, 0}, {4, 150},
                                 {5, 10},  {1, 0},   {3, 0},   {4, 0}, {5, 0}};
    std::unordered_map<uintptr_t, size_t> out;
    inplace_outdata inplace_selection;
    inplace_data inplace_hint = {{4, {{1, InplaceKind::FREE}}}};
    size_t total = memoryplan::scheduleMemoryAllocations(
        traces, 1, false, inplace_hint, out, inplace_selection);
    std::unordered_map<uintptr_t, size_t> expected_out = {
        {1, 0}, {2, 120}, {3, 220}, {4, 0}, {5, 150}};
    EXPECT_EQ(total, 370UL);
    EXPECT_EQ(out, expected_out);

    inplace_outdata expected_inplace = {{4, {1}}};
    EXPECT_EQ(inplace_selection, expected_inplace);
  }

  // perfect matches
  {
    memoryplan::Traces traces = {{1, 120}, {2, 100}, {3, 100}, {4, 120},
                                 {1, 0},   {2, 0},   {3, 0},   {4, 0},
                                 {5, 200}, {5, 0}};
    std::unordered_map<uintptr_t, size_t> out;
    inplace_outdata inplace_selection;
    inplace_data inplace_hint = {
        {3, {{1, InplaceKind::FREE}, {2, InplaceKind::FREE}}},
        {4, {{1, InplaceKind::FREE}, {2, InplaceKind::FREE}}}};
    size_t total = memoryplan::scheduleMemoryAllocations(
        traces, 1, false, inplace_hint, out, inplace_selection);
    std::unordered_map<uintptr_t, size_t> expected_out = {
        {1, 0}, {2, 120}, {3, 120}, {4, 0}, {5, 0}};
    EXPECT_EQ(total, 220UL);
    EXPECT_EQ(out, expected_out);

    inplace_outdata expected_inplace = {{3, {2}}, {4, {1}}};
    EXPECT_EQ(inplace_selection, expected_inplace);
  }

  // selected inputs
  {
    memoryplan::Traces traces = {{1, 120}, {2, 100}, {3, 100}, {4, 120},
                                 {1, 0},   {2, 0},   {3, 0},   {4, 0},
                                 {5, 200}, {5, 0}};
    std::unordered_map<uintptr_t, size_t> out;
    inplace_outdata inplace_selection;
    inplace_data inplace_hint = {{3, {{1, InplaceKind::FREE}}},
                                 {4, {{2, InplaceKind::FREE}}}};
    size_t total = memoryplan::scheduleMemoryAllocations(
        traces, 1, false, inplace_hint, out, inplace_selection);
    std::unordered_map<uintptr_t, size_t> expected_out = {
        {1, 0}, {2, 120}, {3, 0}, {4, 120}, {5, 0}};
    EXPECT_EQ(total, 240UL);
    EXPECT_EQ(out, expected_out);

    inplace_outdata expected_inplace = {{3, {1}}, {4, {2}}};
    EXPECT_EQ(inplace_selection, expected_inplace);
  }
}
