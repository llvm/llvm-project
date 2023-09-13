; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: Entry or anchor intrinsic cannot have a convergencectrl token operand.
; CHECK-NEXT: %t04_tok2 = call token
; CHECK: Loop intrinsic must have a convergencectrl token operand.
; CHECK-NEXT: %t04_tok3 = call token
define void @basic_syntax() {
  %t04_tok1 = call token @llvm.experimental.convergence.anchor()
  %t04_tok2 = call token @llvm.experimental.convergence.anchor() [ "convergencectrl"(token %t04_tok1) ]
  %t04_tok3 = call token @llvm.experimental.convergence.loop()
  ret void
}

; CHECK: Convergence control tokens can only be produced by calls to the convergence control intrinsics.
; CHECK-NEXT:  %t04_tok1 = call token @produce_token()
; CHECK-NEXT:  call void @f() [ "convergencectrl"(token %t04_tok1) ]
define void @wrong_token() {
  %t04_tok1 = call token @produce_token()
  call void @f() [ "convergencectrl"(token %t04_tok1) ]
  ret void
}

; CHECK: Convergence control token can only be used in a convergent call.
; CHECK-NEXT  call void @g(){{.*}}%t05_tok1
define void @missing.attribute() {
  %t05_tok1 = call token @llvm.experimental.convergence.anchor()
  call void @g() [ "convergencectrl"(token %t05_tok1) ]
  ret void
}

; CHECK: The 'convergencectrl' bundle requires exactly one token use.
; CHECK-NEXT:  call void @g()
define void @multiple_tokens() {
  %t06_tok1 = call token @llvm.experimental.convergence.anchor()
  %t06_tok2 = call token @llvm.experimental.convergence.anchor()
  call void @g() [ "convergencectrl"(token %t06_tok2, token %t06_tok1) ]
  ret void
}

; CHECK: The 'convergencectrl' bundle can occur at most once on a call
; CHECK-NEXT:  call void @g()
define void @multiple_bundles() {
  %t07_tok1 = call token @llvm.experimental.convergence.anchor()
  %t07_tok2 = call token @llvm.experimental.convergence.anchor()
  call void @g() [ "convergencectrl"(token %t07_tok2), "convergencectrl"(token %t07_tok1) ]
  ret void
}

; CHECK: Cannot mix controlled and uncontrolled convergence in the same function
; CHECK-NEXT  call void @f()
define void @mixed1() {
  call void @g() ; not convergent
  %t10_tok1 = call token @llvm.experimental.convergence.anchor()
  call void @f() [ "convergencectrl"(token %t10_tok1) ]
  call void @g()
  call void @f() ; uncontrolled convergent
  ret void
}

; CHECK: Cannot mix controlled and uncontrolled convergence in the same function
; CHECK:  %t20_tok1 = call token @llvm.experimental.convergence.anchor()
; CHECK: Cannot mix controlled and uncontrolled convergence in the same function
; CHECK:  call void @f() [ "convergencectrl"(token %t20_tok1) ]
define void @mixed2() {
  call void @g() ; not convergent
  call void @f() ; uncontrolled convergent
  call void @g()
  %t20_tok1 = call token @llvm.experimental.convergence.anchor()
  call void @f() [ "convergencectrl"(token %t20_tok1) ]
  ret void
}

; CHECK: Convergence region is not well-nested.
; CHECK:   %t30_tok2
define void @region_nesting1() {
  %t30_tok1 = call token @llvm.experimental.convergence.anchor()
  %t30_tok2 = call token @llvm.experimental.convergence.anchor()
  call void @f() [ "convergencectrl"(token %t30_tok1) ]
  call void @f() [ "convergencectrl"(token %t30_tok2) ]
  ret void
}

; CHECK: Convergence region is not well-nested.
; CHECK:   %t40_tok2
define void @region_nesting2(i1 %cond) {
A:
  %t40_tok1 = call token @llvm.experimental.convergence.anchor()
  %t40_tok2 = call token @llvm.experimental.convergence.anchor()
  br i1 %cond, label %B, label %C

B:
  call void @f() [ "convergencectrl"(token %t40_tok1) ]
  br label %C

C:
  call void @f() [ "convergencectrl"(token %t40_tok2) ]
  ret void
}

; CHECK: Convergence token used by an instruction other than llvm.experimental.convergence.loop in a cycle that does not contain the token's definition.
; CHECK:   token %t50_tok1
define void @use_in_cycle() {
A:
  %t50_tok1 = call token @llvm.experimental.convergence.anchor()
  br label %B

B:
  call void @f() [ "convergencectrl"(token %t50_tok1) ]
  br label %B
}

; CHECK: Entry intrinsic cannot be preceded by a convergent operation in the same basic block.
; CHECK:   %t60_tok1
define void @entry_at_start(i32 %x, i32 %y) convergent {
  %z = add i32 %x, %y
  call void @f()
  %t60_tok1 = call token @llvm.experimental.convergence.entry()
  ret void
}

; CHECK: Entry intrinsic can occur only in a convergent function.
; CHECK:   %t60_tok2
define void @entry_in_convergent(i32 %x, i32 %y) {
  %t60_tok2 = call token @llvm.experimental.convergence.entry()
  ret void
}

; CHECK: Loop intrinsic cannot be preceded by a convergent operation in the same basic block.
; CHECK-NEXT: %h1
; CHECK-SAME: %t60_tok3
define void @loop_at_start(i32 %x, i32 %y) convergent {
A:
  %t60_tok3 = call token @llvm.experimental.convergence.entry()
  br label %B
B:
  %z = add i32 %x, %y
  ; This is not an error
  %h2 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t60_tok3) ]
  br label %C
C:
  call void @f()
  %h1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t60_tok3) ]
  ret void
}

; CHECK: Entry intrinsic can occur only in the entry block.
; CHECK:   %t60_tok4
define void @entry_at_entry(i32 %x, i32 %y) convergent {
A:
  %z = add i32 %x, %y
  br label %B
B:
  %t60_tok4 = call token @llvm.experimental.convergence.entry()
  ret void
}

; CHECK: Two static convergence token uses in a cycle that does not contain either token's definition.
; CHECK:   token %t70_tok1
; CHECK:   token %t70_tok2
define void @multiple_hearts() {
A:
  %t70_tok1 = call token @llvm.experimental.convergence.anchor()
  %t70_tok2 = call token @llvm.experimental.convergence.anchor()
  br label %B

B:
  %h2 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t70_tok2) ]
  %h1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t70_tok1) ]
  br label %B
}

; CHECK: Two static convergence token uses in a cycle that does not contain either token's definition.
; CHECK:   token %h0
; CHECK:   token %h0
define void @multiple_hearts_nested(i1 %cond1, i1 %cond2) {
A:
  %t70_tok3 = call token @llvm.experimental.convergence.anchor()
  br label %B

B:
  %h0 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t70_tok3) ]
  br i1 %cond1, label %C, label %B

C:
  %h1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %h0) ]
  %h2 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %h0) ]
  br i1 %cond2, label %C, label %B
}

; CHECK: Cycle heart must dominate all blocks in the cycle.
; CHECK: %h3 = call token
; CHECK:   label %C
define void @invalid_heart_nested(i1 %cond1, i1 %cond2) {
A:
  %t70_tok4 = call token @llvm.experimental.convergence.anchor()
  br label %B

B:
  br i1 %cond1, label %C, label %B

C:
  %h3 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t70_tok4) ]
  br i1 %cond2, label %C, label %B
}

; CHECK: Cycle heart must dominate all blocks in the cycle.
; CHECK: %h4 = call token
; CHECK: label %C
define void @irreducible1(i1 %cond) {
A:
  %a = call token @llvm.experimental.convergence.anchor()
  br i1 %cond, label %B, label %C

B:
  %b = call token @llvm.experimental.convergence.anchor()
  br i1 %cond, label %C, label %D

C:
  %h4 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %a) ]
  br i1 %cond, label %B, label %E

D:
  call void @f() [ "convergencectrl"(token %b) ]
  br i1 %cond, label %B, label %F

E:
  call void @f() [ "convergencectrl"(token %h4) ]
  br i1 %cond, label %C, label %F

F:
  call void @f() [ "convergencectrl"(token %a) ]
  ret void
}

; Mirror image of @irreducible1
; CHECK: Cycle heart must dominate all blocks in the cycle.
; CHECK: %h5 = call token
; CHECK: label %B
define void @irreducible2(i1 %cond) {
A:
  %a = call token @llvm.experimental.convergence.anchor()
  br i1 %cond, label %B, label %C

B:
  %h5 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %a) ]
  br i1 %cond, label %C, label %D

C:
  %c = call token @llvm.experimental.convergence.anchor()
  br i1 %cond, label %B, label %E

D:
  call void @f() [ "convergencectrl"(token %h5) ]
  br i1 %cond, label %B, label %F

E:
  call void @f() [ "convergencectrl"(token %c) ]
  br i1 %cond, label %C, label %F

F:
  call void @f() [ "convergencectrl"(token %a) ]
  ret void
}

declare token @produce_token()

declare void @f() convergent
declare void @g()

declare token @llvm.experimental.convergence.entry()
declare token @llvm.experimental.convergence.anchor()
declare token @llvm.experimental.convergence.loop()
