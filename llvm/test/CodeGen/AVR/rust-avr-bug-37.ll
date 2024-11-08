; RUN: llc < %s -march=avr | FileCheck %s

%"fmt::Formatter" = type { i32, { ptr, ptr } }

@str.1b = external constant [0 x i8]

define void @"TryFromIntError::Debug"(ptr dereferenceable(32)) unnamed_addr #0 personality ptr addrspace(1) @rust_eh_personality {
; CHECK-LABEL: "TryFromIntError::Debug"
start:
  %builder = alloca i8, align 8
  %1 = getelementptr inbounds %"fmt::Formatter", ptr %0, i16 0, i32 1
  %2 = bitcast ptr %1 to ptr
  %3 = load ptr, ptr %2, align 2
  %4 = getelementptr inbounds %"fmt::Formatter", ptr %0, i16 0, i32 1, i32 1
  %5 = load ptr, ptr %4, align 2
  %6 = getelementptr inbounds ptr, ptr %5, i16 3
  %7 = bitcast ptr %6 to ptr
  %8 = load ptr addrspace(1), ptr %7, align 2
  %9 = tail call i8 %8(ptr nonnull %3, ptr noalias nonnull readonly @str.1b, i16 15)
  unreachable
}

declare i32 @rust_eh_personality(...) unnamed_addr

attributes #0 = { uwtable }
