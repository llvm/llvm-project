; RUN: llc -verify-machineinstrs -mtriple=x86_64-unknown -x86-lvi-load-dot-verify -o %t < %s | FileCheck %s

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @test(ptr %untrusted_user_ptr, ptr %secret, i32 %secret_size) #0 {
entry:
  %untrusted_user_ptr.addr = alloca ptr, align 8
  %secret.addr = alloca ptr, align 8
  %secret_size.addr = alloca i32, align 4
  %ret_val = alloca i32, align 4
  %i = alloca i32, align 4
  store ptr %untrusted_user_ptr, ptr %untrusted_user_ptr.addr, align 8
  store ptr %secret, ptr %secret.addr, align 8
  store i32 %secret_size, ptr %secret_size.addr, align 4
  store i32 0, ptr %ret_val, align 4
  call void @llvm.x86.sse2.lfence()
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %1 = load i32, ptr %secret_size.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load i32, ptr %i, align 4
  %rem = srem i32 %2, 2
  %cmp1 = icmp eq i32 %rem, 0
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %3 = load ptr, ptr %secret.addr, align 8
  %4 = load i32, ptr %ret_val, align 4
  %idxprom = sext i32 %4 to i64
  %arrayidx = getelementptr inbounds i32, ptr %3, i64 %idxprom
  %5 = load i32, ptr %arrayidx, align 4
  %6 = load ptr, ptr %untrusted_user_ptr.addr, align 8
  store i32 %5, ptr %6, align 4
  br label %if.end

if.else:                                          ; preds = %for.body
  %7 = load ptr, ptr %secret.addr, align 8
  %8 = load i32, ptr %ret_val, align 4
  %idxprom2 = sext i32 %8 to i64
  %arrayidx3 = getelementptr inbounds i32, ptr %7, i64 %idxprom2
  store i32 42, ptr %arrayidx3, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %9 = load ptr, ptr %untrusted_user_ptr.addr, align 8
  %10 = load i32, ptr %9, align 4
  store i32 %10, ptr %ret_val, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %11 = load i32, ptr %i, align 4
  %inc = add nsw i32 %11, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %12 = load i32, ptr %ret_val, align 4
  ret i32 %12
}

; CHECK:      digraph "Speculative gadgets for \"test\" function" {
; CHECK-NEXT: label="Speculative gadgets for \"test\" function";
; CHECK:      Node0x{{[0-9a-f]+}} [shape=record,color = green,label="{LFENCE\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 0];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{renamable $eax = MOV32rm %stack.4.i, 1, $noreg, 0, $noreg :: (dereferenceable load (s32) from %ir.i)\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[color = red, style = "dashed"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{JCC_1 %bb.6, 13, implicit killed $eflags\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{CMP32rm killed renamable $eax, %stack.2.secret_size.addr, 1, $noreg, 0, $noreg, implicit-def $eflags :: (dereferenceable load (s32) from %ir.secret_size.addr)\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[color = red, style = "dashed"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{renamable $eax = MOV32rm %stack.4.i, 1, $noreg, 0, $noreg :: (dereferenceable load (s32) from %ir.i)\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[color = red, style = "dashed"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{JCC_1 %bb.4, 5, implicit killed $eflags\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{renamable $rax = MOV64rm %stack.1.secret.addr, 1, $noreg, 0, $noreg :: (dereferenceable load (s64) from %ir.secret.addr)\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[color = red, style = "dashed"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{renamable $eax = MOV32rm killed renamable $rax, 4, killed renamable $rcx, 0, $noreg :: (load (s32) from %ir.arrayidx)\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{renamable $rcx = MOVSX64rm32 %stack.3.ret_val, 1, $noreg, 0, $noreg :: (dereferenceable load (s32) from %ir.ret_val)\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[color = red, style = "dashed"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{renamable $rcx = MOV64rm %stack.0.untrusted_user_ptr.addr, 1, $noreg, 0, $noreg :: (dereferenceable load (s64) from %ir.untrusted_user_ptr.addr)\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[color = red, style = "dashed"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{MOV32mr killed renamable $rcx, 1, $noreg, 0, $noreg, killed renamable $eax :: (store (s32) into %ir.6)\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{renamable $rax = MOV64rm %stack.1.secret.addr, 1, $noreg, 0, $noreg :: (dereferenceable load (s64) from %ir.secret.addr)\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[color = red, style = "dashed"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{MOV32mi killed renamable $rax, 4, killed renamable $rcx, 0, $noreg, 42 :: (store (s32) into %ir.arrayidx3)\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{renamable $rcx = MOVSX64rm32 %stack.3.ret_val, 1, $noreg, 0, $noreg :: (dereferenceable load (s32) from %ir.ret_val)\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[color = red, style = "dashed"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{renamable $rax = MOV64rm %stack.0.untrusted_user_ptr.addr, 1, $noreg, 0, $noreg :: (dereferenceable load (s64) from %ir.untrusted_user_ptr.addr)\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[color = red, style = "dashed"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{renamable $eax = MOV32rm killed renamable $rax, 1, $noreg, 0, $noreg :: (load (s32) from %ir.9)\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,color = blue,label="{ARGS}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 0];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{MOV64mr %stack.0.untrusted_user_ptr.addr, 1, $noreg, 0, $noreg, killed renamable $rdi :: (store (s64) into %ir.untrusted_user_ptr.addr)\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 0];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{JMP_1 %bb.5\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{JMP_1 %bb.1\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 1];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{renamable $eax = MOV32rm %stack.3.ret_val, 1, $noreg, 0, $noreg :: (dereferenceable load (s32) from %ir.ret_val)\n}"];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label = 0];
; CHECK-NEXT: Node0x{{[0-9a-f]+}} [shape=record,label="{RET 0, $eax\n}"];
; CHECK-NEXT: }

; Function Attrs: nounwind
declare void @llvm.x86.sse2.lfence() #1

attributes #0 = { "target-features"="+lvi-cfi"
                  "target-features"="+lvi-load-hardening" }
attributes #1 = { nounwind }
