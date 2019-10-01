//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Automatically generated file, please consult code owner before editing.
//===----------------------------------------------------------------------===//

def ArchV66: SubtargetFeature<"v66", "HexagonArchVersion", "Hexagon::ArchEnum::V66", "Enable Hexagon V66 architecture">;
def HasV66 : Predicate<"HST->hasV66Ops()">, AssemblerPredicate<"ArchV66">;
def ArchV65: SubtargetFeature<"v65", "HexagonArchVersion", "Hexagon::ArchEnum::V65", "Enable Hexagon V65 architecture">;
def HasV65 : Predicate<"HST->hasV65Ops()">, AssemblerPredicate<"ArchV65">;
def ArchV62: SubtargetFeature<"v62", "HexagonArchVersion", "Hexagon::ArchEnum::V62", "Enable Hexagon V62 architecture">;
def HasV62 : Predicate<"HST->hasV62Ops()">, AssemblerPredicate<"ArchV62">;
def ArchV60: SubtargetFeature<"v60", "HexagonArchVersion", "Hexagon::ArchEnum::V60", "Enable Hexagon V60 architecture">;
def HasV60 : Predicate<"HST->hasV60Ops()">, AssemblerPredicate<"ArchV60">;
def ArchV55: SubtargetFeature<"v55", "HexagonArchVersion", "Hexagon::ArchEnum::V55", "Enable Hexagon V55 architecture">;
def HasV55 : Predicate<"HST->hasV55Ops()">, AssemblerPredicate<"ArchV55">;
def ArchV5: SubtargetFeature<"v5", "HexagonArchVersion", "Hexagon::ArchEnum::V5", "Enable Hexagon V5 architecture">;
def HasV5 : Predicate<"HST->hasV5Ops()">, AssemblerPredicate<"ArchV5">;
