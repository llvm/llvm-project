; Make sure a profile that is generated from a function without an exit node
; does not cause an assertion. The profile consists of a non-zero count in a
; basic block and 0 counts in all succcessor blocks. Expect a warning.

; RUN: llvm-profdata merge %S/Inputs/maxcountzero.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-instrument-entry=false -pgo-test-profile-file=%t.profdata -S 2>&1 | FileCheck %s

define void @bar(i32 noundef %s) {
entry:
  %cmp = icmp sgt i32 %s, 20
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void @exit(i32 noundef 1)
  unreachable

if.end:
  ret void
}

declare void @exit(i32 noundef)

define void @foo(i32 noundef %n) {
entry:
  %sum = alloca i32, align 4
  store volatile i32 %n, ptr %sum, align 4
  %sum.0.sum.0. = load volatile i32, ptr %sum, align 4
  call void @bar(i32 noundef %sum.0.sum.0.)
  %cmp = icmp slt i32 %n, 10
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %sum.0.sum.0.1 = load volatile i32, ptr %sum, align 4
  call void @bar(i32 noundef %sum.0.sum.0.1)
  br label %if.end

if.end:
  br label %for.cond

for.cond:
  %sum.0.sum.0.2 = load volatile i32, ptr %sum, align 4
  call void @bar(i32 noundef %sum.0.sum.0.2)
  br label %for.cond
}

; CHECK: warning:{{.*}}Profile in foo partially ignored
; CHECK: define
