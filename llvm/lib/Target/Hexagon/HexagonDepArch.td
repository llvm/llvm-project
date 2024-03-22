//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Automatically generated file, do not edit!
//===----------------------------------------------------------------------===//

def ArchV5 : SubtargetFeature<"v5", "HexagonArchVersion", "Hexagon::ArchEnum::V5", "Enable Hexagon V5 architecture">;
def HasV5  : Predicate<"HST->hasFeature(llvm::Hexagon::ArchV5)">, AssemblerPredicate<(all_of ArchV5)>;
def ArchV55: SubtargetFeature<"v55", "HexagonArchVersion", "Hexagon::ArchEnum::V55", "Enable Hexagon V55 architecture", [ArchV5]>;
def HasV55 : Predicate<"HST->hasFeature(llvm::Hexagon::ArchV55)">, AssemblerPredicate<(all_of ArchV55)>;
def ArchV60: SubtargetFeature<"v60", "HexagonArchVersion", "Hexagon::ArchEnum::V60", "Enable Hexagon V60 architecture", [ArchV55]>;
def HasV60 : Predicate<"HST->hasFeature(llvm::Hexagon::ArchV60)">, AssemblerPredicate<(all_of ArchV60)>;
def ArchV62: SubtargetFeature<"v62", "HexagonArchVersion", "Hexagon::ArchEnum::V62", "Enable Hexagon V62 architecture", [ArchV60]>;
def HasV62 : Predicate<"HST->hasFeature(llvm::Hexagon::ArchV62)">, AssemblerPredicate<(all_of ArchV62)>;
def ArchV65: SubtargetFeature<"v65", "HexagonArchVersion", "Hexagon::ArchEnum::V65", "Enable Hexagon V65 architecture", [ArchV62]>;
def HasV65 : Predicate<"HST->hasFeature(llvm::Hexagon::ArchV65)">, AssemblerPredicate<(all_of ArchV65)>;
def ArchV66: SubtargetFeature<"v66", "HexagonArchVersion", "Hexagon::ArchEnum::V66", "Enable Hexagon V66 architecture", [ArchV65]>;
def HasV66 : Predicate<"HST->hasFeature(llvm::Hexagon::ArchV66)">, AssemblerPredicate<(all_of ArchV66)>;
def ArchV67: SubtargetFeature<"v67", "HexagonArchVersion", "Hexagon::ArchEnum::V67", "Enable Hexagon V67 architecture", [ArchV66]>;
def HasV67 : Predicate<"HST->hasFeature(llvm::Hexagon::ArchV67)">, AssemblerPredicate<(all_of ArchV67)>;
def ArchV68: SubtargetFeature<"v68", "HexagonArchVersion", "Hexagon::ArchEnum::V68", "Enable Hexagon V68 architecture", [ArchV67]>;
def HasV68 : Predicate<"HST->hasFeature(llvm::Hexagon::ArchV68)">, AssemblerPredicate<(all_of ArchV68)>;
def ArchV69: SubtargetFeature<"v69", "HexagonArchVersion", "Hexagon::ArchEnum::V69", "Enable Hexagon V69 architecture", [ArchV68]>;
def HasV69 : Predicate<"HST->hasFeature(llvm::Hexagon::ArchV69)">, AssemblerPredicate<(all_of ArchV69)>;
def ArchV71: SubtargetFeature<"v71", "HexagonArchVersion", "Hexagon::ArchEnum::V71", "Enable Hexagon V71 architecture", [ArchV69]>;
def HasV71 : Predicate<"HST->hasFeature(llvm::Hexagon::ArchV71)">, AssemblerPredicate<(all_of ArchV71)>;
def ArchV73: SubtargetFeature<"v73", "HexagonArchVersion", "Hexagon::ArchEnum::V73", "Enable Hexagon V73 architecture", [ArchV71]>;
def HasV73 : Predicate<"HST->hasFeature(llvm::Hexagon::ArchV73)">, AssemblerPredicate<(all_of ArchV73)>;
