//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Automatically generated file, do not edit!
//===----------------------------------------------------------------------===//

def ExtensionHVXV60 : SubtargetFeature<"hvxv60", "HexagonHVXVersion", "Hexagon::ArchEnum::V60", "Hexagon HVX V60 instructions", [ExtensionHVX]>;
def UseHVXV60       : Predicate<"HST->hasFeature(llvm::Hexagon::ExtensionHVXV60)">, AssemblerPredicate<(all_of ExtensionHVXV60)>;
def ExtensionHVXV62 : SubtargetFeature<"hvxv62", "HexagonHVXVersion", "Hexagon::ArchEnum::V62", "Hexagon HVX V62 instructions", [ExtensionHVXV60]>;
def UseHVXV62       : Predicate<"HST->hasFeature(llvm::Hexagon::ExtensionHVXV62)">, AssemblerPredicate<(all_of ExtensionHVXV62)>;
def ExtensionHVXV65 : SubtargetFeature<"hvxv65", "HexagonHVXVersion", "Hexagon::ArchEnum::V65", "Hexagon HVX V65 instructions", [ExtensionHVXV62]>;
def UseHVXV65       : Predicate<"HST->hasFeature(llvm::Hexagon::ExtensionHVXV65)">, AssemblerPredicate<(all_of ExtensionHVXV65)>;
def ExtensionHVXV66 : SubtargetFeature<"hvxv66", "HexagonHVXVersion", "Hexagon::ArchEnum::V66", "Hexagon HVX V66 instructions", [ExtensionHVXV65, ExtensionZReg]>;
def UseHVXV66       : Predicate<"HST->hasFeature(llvm::Hexagon::ExtensionHVXV66)">, AssemblerPredicate<(all_of ExtensionHVXV66)>;
def ExtensionHVXV67 : SubtargetFeature<"hvxv67", "HexagonHVXVersion", "Hexagon::ArchEnum::V67", "Hexagon HVX V67 instructions", [ExtensionHVXV66]>;
def UseHVXV67       : Predicate<"HST->hasFeature(llvm::Hexagon::Arch67)">, AssemblerPredicate<(all_of ExtensionHVXV67)>;
def ExtensionHVXV68 : SubtargetFeature<"hvxv68", "HexagonHVXVersion", "Hexagon::ArchEnum::V68", "Hexagon HVX V68 instructions", [ExtensionHVXV67]>;
def UseHVXV68       : Predicate<"HST->hasFeature(llvm::Hexagon::ExtensionHVXV68)">, AssemblerPredicate<(all_of ExtensionHVXV68)>;
def ExtensionHVXV69 : SubtargetFeature<"hvxv69", "HexagonHVXVersion", "Hexagon::ArchEnum::V69", "Hexagon HVX V69 instructions", [ExtensionHVXV68]>;
def UseHVXV69       : Predicate<"HST->hasFeature(llvm::Hexagon::ExtensionHVXV69)">, AssemblerPredicate<(all_of ExtensionHVXV69)>;
def ExtensionHVXV71 : SubtargetFeature<"hvxv71", "HexagonHVXVersion", "Hexagon::ArchEnum::V71", "Hexagon HVX V71 instructions", [ExtensionHVXV69]>;
def UseHVXV71       : Predicate<"HST->hasFeature(llvm::Hexagon::ExtensionHVXV71)">, AssemblerPredicate<(all_of ExtensionHVXV71)>;
def ExtensionHVXV73 : SubtargetFeature<"hvxv73", "HexagonHVXVersion", "Hexagon::ArchEnum::V73", "Hexagon HVX V73 instructions", [ExtensionHVXV71]>;
def UseHVXV73       : Predicate<"HST->hasFeature(llvm::Hexagon::ExtensionHVXV73)">, AssemblerPredicate<(all_of ExtensionHVXV73)>;
