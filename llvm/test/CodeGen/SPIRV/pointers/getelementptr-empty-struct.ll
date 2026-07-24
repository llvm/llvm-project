; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpFunction

%empty = type {}

@global = internal global %empty zeroinitializer
@in_idx = internal global i32 zeroinitializer

define void @foo() {
entry:
  %idx = load i32, ptr @in_idx
  %gep = getelementptr inbounds i8, ptr @global, i32 %idx
  ret void
}
