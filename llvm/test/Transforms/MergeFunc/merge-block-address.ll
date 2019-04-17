; RUN: opt -S -mergefunc < %s | FileCheck %s

; These two functions are identical. The basic block labels are the same, and
; induce the same CFG. We are testing that block addresses within different
; functions are compared by their value, and not based on order. Both functions
; come from the same C-code, but in the first the two val_0/val_1 basic blocks
; are in a different order (they were manually switched post-compilation).

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z1fi(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  %ret = alloca i32, align 4
  %l = alloca i8*, align 8
  store i32 %i, i32* %i.addr, align 4
  store i32 0, i32* %ret, align 4
  store i8* blockaddress(@_Z1fi, %val_0), i8** %l, align 8
  %0 = load i32, i32* %i.addr, align 4
  %and = and i32 %0, 256
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i8* blockaddress(@_Z1fi, %val_1), i8** %l, align 8
  br label %if.end

if.end:
  %1 = load i8*, i8** %l, align 8
  br label %indirectgoto

val_1:
  store i32 42, i32* %ret, align 4
  br label %end

val_0:
  store i32 12, i32* %ret, align 4
  br label %end


end:
  %2 = load i32, i32* %ret, align 4
  ret i32 %2

indirectgoto:
  %indirect.goto.dest = phi i8* [ %1, %if.end ]
  indirectbr i8* %indirect.goto.dest, [label %val_0, label %val_1]
}

define i32 @_Z1gi(i32 %i) #0 {
; CHECK-LABEL: define i32 @_Z1gi
; CHECK-NEXT: tail call i32 @_Z1fi
; CHECK-NEXT: ret
entry:
  %i.addr = alloca i32, align 4
  %ret = alloca i32, align 4
  %l = alloca i8*, align 8
  store i32 %i, i32* %i.addr, align 4
  store i32 0, i32* %ret, align 4
  store i8* blockaddress(@_Z1gi, %val_0), i8** %l, align 8
  %0 = load i32, i32* %i.addr, align 4
  %and = and i32 %0, 256
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i8* blockaddress(@_Z1gi, %val_1), i8** %l, align 8
  br label %if.end

if.end:
  %1 = load i8*, i8** %l, align 8
  br label %indirectgoto

val_0:
  store i32 12, i32* %ret, align 4
  br label %end

val_1:
  store i32 42, i32* %ret, align 4
  br label %end

end:
  %2 = load i32, i32* %ret, align 4
  ret i32 %2

indirectgoto:
  %indirect.goto.dest = phi i8* [ %1, %if.end ]
  indirectbr i8* %indirect.goto.dest, [label %val_0, label %val_1]
}

