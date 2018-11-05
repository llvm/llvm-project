; RUN: llc < %s -march=avr | FileCheck %s

%"fmt::Formatter" = type { i32, { i8*, void (i8*)** } }

@str.1b = external constant [0 x i8]

define void @"TryFromIntError::Debug"(%"fmt::Formatter"* dereferenceable(32)) unnamed_addr #0 personality i32 (...)* @rust_eh_personality {
; CHECK-LABEL: "TryFromIntError::Debug"
start:
  %builder = alloca i8, align 8
  %1 = getelementptr inbounds %"fmt::Formatter", %"fmt::Formatter"* %0, i16 0, i32 1
  %2 = bitcast { i8*, void (i8*)** }* %1 to {}**
  %3 = load {}*, {}** %2, align 2
  %4 = getelementptr inbounds %"fmt::Formatter", %"fmt::Formatter"* %0, i16 0, i32 1, i32 1
  %5 = load void (i8*)**, void (i8*)*** %4, align 2
  %6 = getelementptr inbounds void (i8*)*, void (i8*)** %5, i16 3
  %7 = bitcast void (i8*)** %6 to i8 ({}*, i8*, i16)**
  %8 = load i8 ({}*, i8*, i16)*, i8 ({}*, i8*, i16)** %7, align 2
  %9 = tail call i8 %8({}* nonnull %3, i8* noalias nonnull readonly getelementptr inbounds ([0 x i8], [0 x i8]* @str.1b, i16 0, i16 0), i16 15)
  unreachable
}

declare i32 @rust_eh_personality(...) unnamed_addr

attributes #0 = { uwtable }