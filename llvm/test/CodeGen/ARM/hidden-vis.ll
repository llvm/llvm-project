; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin | \
; RUN:   grep .private_extern | count 2

%struct.Person = type { i32 }
@a = hidden global i32 0
@b = external global i32


define weak hidden void @_ZN6Person13privateMethodEv(%struct.Person* %this) {
  ret void
}

declare void @function(i32)

define weak void @_ZN6PersonC1Ei(%struct.Person* %this, i32 %_c) {
  ret void
}

