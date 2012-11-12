; RUN: llc %s -o -

; ModuleID = 'undo.c'
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-freebsd"

%struct.__sFILE = type {}
%struct.pos_T = type { i64 }

; check that we're not copying stuff between R and X registers
define internal void @serialize_pos(%struct.pos_T* byval %pos, %struct.__sFILE* %fp) nounwind {
entry:
  ret void
}
