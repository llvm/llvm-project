; RUN: llc < %s -mtriple=i686-windows | FileCheck %s -check-prefix=NORMAL
; RUN: llc < %s -mtriple=i686-windows -no-x86-call-frame-opt | FileCheck %s -check-prefix=NOPUSH
; RUN: llc < %s -mtriple=x86_64-windows | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mtriple=i686-windows -stackrealign -stack-alignment=32 | FileCheck %s -check-prefix=ALIGNED
; RUN: llc < %s -mtriple=i686-pc-linux | FileCheck %s -check-prefix=LINUX

%class.Class = type { i32 }
%struct.s = type { i64 }

declare void @good(i32 %a, i32 %b, i32 %c, i32 %d)
declare void @inreg(i32 %a, i32 inreg %b, i32 %c, i32 %d)
declare x86_thiscallcc void @thiscall(%class.Class* %class, i32 %a, i32 %b, i32 %c, i32 %d)
declare void @oneparam(i32 %a)
declare void @eightparams(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h)
declare void @eightparams16(i16 %a, i16 %b, i16 %c, i16 %d, i16 %e, i16 %f, i16 %g, i16 %h)
declare void @eightparams64(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h)
declare void @struct(%struct.s* byval(%struct.s) %a, i32 %b, i32 %c, i32 %d)
declare void @inalloca(<{ %struct.s }>* inalloca(<{ %struct.s }>))

declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8*)

; We should get pushes for x86, even though there is a reserved call frame.
; Make sure we don't touch x86-64, and that turning it off works.
; NORMAL-LABEL: test1:
; NORMAL-NOT: subl {{.*}} %esp
; NORMAL: pushl   $4
; NORMAL-NEXT: pushl   $3
; NORMAL-NEXT: pushl   $2
; NORMAL-NEXT: pushl   $1
; NORMAL-NEXT: call
; NORMAL-NEXT: addl $16, %esp
; X64-LABEL: test1:
; X64: movl    $1, %ecx
; X64-NEXT: movl    $2, %edx
; X64-NEXT: movl    $3, %r8d
; X64-NEXT: movl    $4, %r9d
; X64-NEXT: callq   good
; NOPUSH-LABEL: test1:
; NOPUSH: subl    $16, %esp
; NOPUSH-NEXT: movl    $4, 12(%esp)
; NOPUSH-NEXT: movl    $3, 8(%esp)
; NOPUSH-NEXT: movl    $2, 4(%esp)
; NOPUSH-NEXT: movl    $1, (%esp)
; NOPUSH-NEXT: call
; NOPUSH-NEXT: addl $16, %esp
define void @test1() {
entry:
  call void @good(i32 1, i32 2, i32 3, i32 4)
  ret void
}

; If we have a reserved frame, we should have pushes
; NORMAL-LABEL: test2:
; NORMAL-NOT: subl {{.*}} %esp
; NORMAL: pushl   $4
; NORMAL-NEXT: pushl   $3
; NORMAL-NEXT: pushl   $2
; NORMAL-NEXT: pushl   $1
; NORMAL-NEXT: call
define void @test2(i32 %k) {
entry:
  %a = alloca i32, i32 %k
  call void @good(i32 1, i32 2, i32 3, i32 4)
  ret void
}

; Again, we expect a sequence of 4 immediate pushes
; Checks that we generate the right pushes for >8bit immediates
; NORMAL-LABEL: test2b:
; NORMAL-NOT: subl {{.*}} %esp
; NORMAL: pushl   $4096
; NORMAL-NEXT: pushl   $3072
; NORMAL-NEXT: pushl   $2048
; NORMAL-NEXT: pushl   $1024
; NORMAL-NEXT: call
; NORMAL-NEXT: addl $16, %esp
define void @test2b() optsize {
entry:
  call void @good(i32 1024, i32 2048, i32 3072, i32 4096)
  ret void
}

; The first push should push a register
; NORMAL-LABEL: test3:
; NORMAL-NOT: subl {{.*}} %esp
; NORMAL: pushl   $4
; NORMAL-NEXT: pushl   $3
; NORMAL-NEXT: pushl   $2
; NORMAL-NEXT: pushl   %e{{..}}
; NORMAL-NEXT: call
; NORMAL-NEXT: addl $16, %esp
define void @test3(i32 %k) optsize {
entry:
  %f = add i32 %k, 1
  call void @good(i32 %f, i32 2, i32 3, i32 4)
  ret void
}

; We support weird calling conventions
; NORMAL-LABEL: test4:
; NORMAL: movl    $2, %eax
; NORMAL-NEXT: pushl   $4
; NORMAL-NEXT: pushl   $3
; NORMAL-NEXT: pushl   $1
; NORMAL-NEXT: call
; NORMAL-NEXT: addl $12, %esp
define void @test4() optsize {
entry:
  call void @inreg(i32 1, i32 2, i32 3, i32 4)
  ret void
}

; NORMAL-LABEL: test4b:
; NORMAL: movl 4(%esp), %ecx
; NORMAL-NEXT: pushl   $4
; NORMAL-NEXT: pushl   $3
; NORMAL-NEXT: pushl   $2
; NORMAL-NEXT: pushl   $1
; NORMAL-NEXT: call
; NORMAL-NEXT: ret
define void @test4b(%class.Class* %f) optsize {
entry:
  call x86_thiscallcc void @thiscall(%class.Class* %f, i32 1, i32 2, i32 3, i32 4)
  ret void
}

; When there is no reserved call frame, check that additional alignment
; is added when the pushes don't add up to the required alignment.
; ALIGNED-LABEL: test5:
; ALIGNED: subl    $16, %esp
; ALIGNED-NEXT: pushl   $4
; ALIGNED-NEXT: pushl   $3
; ALIGNED-NEXT: pushl   $2
; ALIGNED-NEXT: pushl   $1
; ALIGNED-NEXT: call
define void @test5(i32 %k) {
entry:
  %a = alloca i32, i32 %k
  call void @good(i32 1, i32 2, i32 3, i32 4)
  ret void
}

; When the alignment adds up, do the transformation
; ALIGNED-LABEL: test5b:
; ALIGNED: pushl   $8
; ALIGNED-NEXT: pushl   $7
; ALIGNED-NEXT: pushl   $6
; ALIGNED-NEXT: pushl   $5
; ALIGNED-NEXT: pushl   $4
; ALIGNED-NEXT: pushl   $3
; ALIGNED-NEXT: pushl   $2
; ALIGNED-NEXT: pushl   $1
; ALIGNED-NEXT: call
define void @test5b() optsize {
entry:
  call void @eightparams(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8)
  ret void
}

; When having to compensate for the alignment isn't worth it,
; don't use pushes.
; ALIGNED-LABEL: test5c:
; ALIGNED: movl $1, (%esp)
; ALIGNED-NEXT: call
define void @test5c() optsize {
entry:
  call void @oneparam(i32 1)
  ret void
}

; Check that pushing the addresses of globals (Or generally, things that
; aren't exactly immediates) isn't broken.
; Fixes PR21878.
; NORMAL-LABEL: test6:
; NORMAL: pushl    $_ext
; NORMAL-NEXT: call
declare void @f(i8*)
@ext = external dso_local constant i8

define void @test6() {
  call void @f(i8* @ext)
  br label %bb
bb:
  alloca i32
  ret void
}

; Check that we fold simple cases into the push
; NORMAL-LABEL: test7:
; NORMAL-NOT: subl {{.*}} %esp
; NORMAL: movl 4(%esp), [[EAX:%e..]]
; NORMAL-NEXT: pushl   $4
; NORMAL-NEXT: pushl   ([[EAX]])
; NORMAL-NEXT: pushl   $2
; NORMAL-NEXT: pushl   $1
; NORMAL-NEXT: call
; NORMAL-NEXT: addl $16, %esp
define void @test7(i32* %ptr) optsize {
entry:
  %val = load i32, i32* %ptr
  call void @good(i32 1, i32 2, i32 %val, i32 4)
  ret void
}

; Fold stack-relative loads into the push, with correct offset
; In particular, at the second push, %b was at 12(%esp) and
; %a wast at 8(%esp), but the second push bumped %esp, so %a
; is now it at 12(%esp)
; NORMAL-LABEL: test8:
; NORMAL: pushl   $4
; NORMAL-NEXT: pushl   12(%esp)
; NORMAL-NEXT: pushl   12(%esp)
; NORMAL-NEXT: pushl   $1
; NORMAL-NEXT: call
; NORMAL-NEXT: addl $16, %esp
define void @test8(i32 %a, i32 %b) optsize {
entry:
  call void @good(i32 1, i32 %a, i32 %b, i32 4)
  ret void
}

; If one function is using push instructions, and the other isn't
; (because it has frame-index references), then we must resolve
; these references correctly.
; NORMAL-LABEL: test9:
; NORMAL-NOT: leal (%esp),
; NORMAL: pushl $4
; NORMAL-NEXT: pushl $3
; NORMAL-NEXT: pushl $2
; NORMAL-NEXT: pushl $1
; NORMAL-NEXT: call
; NORMAL-NEXT: addl $16, %esp
; NORMAL-NEXT: movl (%esp), [[E1:%e..]]
; NORMAL-NEXT: movl 4(%esp), [[E2:%e..]]
; NORMAL-NEXT: leal 16(%esp), [[E3:%e..]]
; NORMAL-NEXT: leal 12(%esp), [[E4:%e..]]
; NORMAL-NEXT: pushl   [[E3]]
; NORMAL-NEXT: pushl   [[E4]]
; NORMAL-NEXT: pushl   $6
; NORMAL-NEXT: pushl   [[E2]]
; NORMAL-NEXT: pushl   [[E1]]
; NORMAL-NEXT: call
; NORMAL-NEXT: addl $20, %esp
define void @test9() optsize {
entry:
  %p = alloca i32, align 4
  %q = alloca i32, align 4
  %s = alloca %struct.s, align 8
  call void @good(i32 1, i32 2, i32 3, i32 4)
  %pv = ptrtoint i32* %p to i32
  %qv = ptrtoint i32* %q to i32
  call void @struct(%struct.s* byval(%struct.s) %s, i32 6, i32 %qv, i32 %pv)
  ret void
}

; We can end up with an indirect call which gets reloaded on the spot.
; Make sure we reference the correct stack slot - we spill into (%esp)
; and reload from 16(%esp) due to the pushes.
; NORMAL-LABEL: test10:
; NORMAL: movl $_good, [[ALLOC:.*]]
; NORMAL-NEXT: movl [[ALLOC]], [[EAX:%e..]]
; NORMAL-NEXT: movl [[EAX]], (%esp) # 4-byte Spill
; NORMAL: nop
; NORMAL: pushl $4
; NORMAL-NEXT: pushl $3
; NORMAL-NEXT: pushl $2
; NORMAL-NEXT: pushl $1
; NORMAL-NEXT: calll *16(%esp)
; NORMAL-NEXT: addl $24, %esp
define void @test10() optsize {
  %stack_fptr = alloca void (i32, i32, i32, i32)*
  store void (i32, i32, i32, i32)* @good, void (i32, i32, i32, i32)** %stack_fptr
  %good_ptr = load volatile void (i32, i32, i32, i32)*, void (i32, i32, i32, i32)** %stack_fptr
  call void asm sideeffect "nop", "~{ax},~{bx},~{cx},~{dx},~{bp},~{si},~{di}"()
  call void (i32, i32, i32, i32) %good_ptr(i32 1, i32 2, i32 3, i32 4)
  ret void
}

; We can't fold the load from the global into the push because of
; interference from the store
; NORMAL-LABEL: test11:
; NORMAL: movl    _the_global, [[EAX:%e..]]
; NORMAL-NEXT: movl    $42, _the_global
; NORMAL-NEXT: pushl $4
; NORMAL-NEXT: pushl $3
; NORMAL-NEXT: pushl $2
; NORMAL-NEXT: pushl [[EAX]]
; NORMAL-NEXT: call
; NORMAL-NEXT: addl $16, %esp
@the_global = external dso_local global i32
define void @test11() optsize {
  %myload = load i32, i32* @the_global
  store i32 42, i32* @the_global
  call void @good(i32 %myload, i32 2, i32 3, i32 4)
  ret void
}

; Converting one mov into a push isn't worth it when
; doing so forces too much overhead for other calls.
; NORMAL-LABEL: test12:
; NORMAL:       pushl  $8
; NORMAL-NEXT:  pushl  $7
; NORMAL-NEXT:  pushl  $6
; NORMAL-NEXT:  pushl  $5
; NORMAL-NEXT: calll _good
define void @test12() optsize {
entry:
  %s = alloca %struct.s, align 4
  call void @struct(%struct.s* %s, i32 2, i32 3, i32 4)
  call void @good(i32 5, i32 6, i32 7, i32 8)
  call void @struct(%struct.s* %s, i32 10, i32 11, i32 12)
  ret void
}

; But if the gains outweigh the overhead, we should do it
; NORMAL-LABEL: test12b:
; NORMAL: pushl    $4
; NORMAL-NEXT: pushl    $3
; NORMAL-NEXT: pushl    $2
; NORMAL-NEXT: pushl    $1
; NORMAL-NEXT: calll _good
; NORMAL-NEXT: addl    $16, %esp
; NORMAL=NEXT: movl  (%esp), %eax
; NORMAL=NEXT: movl  4(%esp), %ecx
; NORMAL=NEXT: pushl  $8
; NORMAL=NEXT: pushl  $7
; NORMAL=NEXT: pushl  $6
; NORMAL=NEXT: pushl  %ecx
; NORMAL=NEXT: pushl  %eax
; NORMAL=NEXT: calll  _struct
; NORMAL=NEXT: addl  $20, %esp
; NORMAL=NEXT: pushl  $12
; NORMAL=NEXT: pushl  $11
; NORMAL=NEXT: pushl  $10
; NORMAL=NEXT: pushl  $9
; NORMAL=NEXT: calll  _good
; NORMAL=NEXT: addl  $16, %esp
define void @test12b() optsize {
entry:
  %s = alloca %struct.s, align 4
  call void @good(i32 1, i32 2, i32 3, i32 4)
  call void @struct(%struct.s* %s, i32 6, i32 7, i32 8)
  call void @good(i32 9, i32 10, i32 11, i32 12)
  ret void
}

; Make sure the add does not prevent folding loads into pushes.
; val1 and val2 will not be folded into pushes since they have
; an additional use, but val3 should be.
; NORMAL-LABEL: test13:
; NORMAL: movl ([[P1:%e..]]), [[V1:%e..]]
; NORMAL-NEXT: movl ([[P2:%e..]]), [[V2:%e..]]
; NORMAL-NEXT: , [[ADD:%e..]]
; NORMAL-NEXT: pushl [[ADD]]
; NORMAL-NEXT: pushl ([[P3:%e..]])
; NORMAL-NEXT: pushl [[V2]]
; NORMAL-NEXT: pushl [[V1]]
; NORMAL-NEXT: calll _good
; NORMAL: movl [[P3]], %eax
define i32* @test13(i32* inreg %ptr1, i32* inreg %ptr2, i32* inreg %ptr3) optsize {
entry:
  %val1 = load i32, i32* %ptr1
  %val2 = load i32, i32* %ptr2
  %val3 = load i32, i32* %ptr3
  %add = add i32 %val1, %val2
  call void @good(i32 %val1, i32 %val2, i32 %val3, i32 %add)
  ret i32* %ptr3
}

; Make sure to fold adjacent stack adjustments.
; LINUX-LABEL: pr27140:
; LINUX: subl    $12, %esp
; LINUX: .cfi_def_cfa_offset 16
; LINUX-NOT: sub
; LINUX: pushl   $4
; LINUX: .cfi_adjust_cfa_offset 4
; LINUX: pushl   $3
; LINUX: .cfi_adjust_cfa_offset 4
; LINUX: pushl   $2
; LINUX: .cfi_adjust_cfa_offset 4
; LINUX: pushl   $1
; LINUX: .cfi_adjust_cfa_offset 4
; LINUX: calll   good
; LINUX: addl    $28, %esp
; LINUX: .cfi_adjust_cfa_offset -28
; LINUX-NOT: add
; LINUX: retl
define void @pr27140() optsize {
entry:
  tail call void @good(i32 1, i32 2, i32 3, i32 4)
  ret void
}

; Check that a stack restore (leal -4(%ebp), %esp) doesn't get merged with a
; stack adjustment (addl $12, %esp). Just because it's a lea doesn't mean it's
; simply decreasing the stack pointer.
; NORMAL-LABEL: test14:
; NORMAL: calll _B_func
; NORMAL: leal -4(%ebp), %esp
; NORMAL-NOT: %esp
; NORMAL: retl
%struct.A = type { i32, i32 }
%struct.B = type { i8 }
declare x86_thiscallcc %struct.B* @B_ctor(%struct.B* returned, %struct.A* byval(%struct.A))
declare void @B_func(%struct.B* sret(%struct.B), %struct.B*, i32)
define void @test14(%struct.A* %a) {
entry:
  %ref.tmp = alloca %struct.B, align 1
  %agg.tmp = alloca i64, align 8
  %tmpcast = bitcast i64* %agg.tmp to %struct.A*
  %tmp = alloca %struct.B, align 1
  %0 = bitcast %struct.A* %a to i64*
  %1 = load i64, i64* %0, align 4
  store i64 %1, i64* %agg.tmp, align 4
  %call = call x86_thiscallcc %struct.B* @B_ctor(%struct.B* %ref.tmp, %struct.A* byval(%struct.A) %tmpcast)
  %2 = getelementptr inbounds %struct.B, %struct.B* %tmp, i32 0, i32 0
  call void @B_func(%struct.B* sret(%struct.B) %tmp, %struct.B* %ref.tmp, i32 1)
  ret void
}

; NORMAL-LABEL: pr34863_16
; NORMAL:       movl  4(%esp), %eax
; NORMAL-NEXT:  pushl  $65535
; NORMAL-NEXT:  pushl  $0
; NORMAL-NEXT:  pushl  %eax
; NORMAL-NEXT:  pushl  %eax
; NORMAL-NEXT:  pushl  %eax
; NORMAL-NEXT:  pushl  %eax
; NORMAL-NEXT:  pushl  %eax
; NORMAL-NEXT:  pushl  %eax
; NORMAL-NEXT:  calll  _eightparams16
; NORMAL-NEXT:  addl  $32, %esp
;
; NOPUSH-LABEL: pr34863_16
; NOPUSH:       subl  $32, %esp
; NOPUSH-NEXT:  movl  36(%esp), %eax
; NOPUSH-NEXT:  movl  %eax, 20(%esp)
; NOPUSH-NEXT:  movl  %eax, 16(%esp)
; NOPUSH-NEXT:  movl  %eax, 12(%esp)
; NOPUSH-NEXT:  movl  %eax, 8(%esp)
; NOPUSH-NEXT:  movl  %eax, 4(%esp)
; NOPUSH-NEXT:  movl  %eax, (%esp)
; NOPUSH-NEXT:  movl  $65535, 28(%esp)
; NOPUSH-NEXT:  andl  $0, 24(%esp)
; NOPUSH-NEXT:  calll  _eightparams16
; NOPUSH-NEXT:   addl  $32, %esp
define void @pr34863_16(i16 %x) minsize nounwind {
entry:
  tail call void @eightparams16(i16 %x, i16 %x, i16 %x, i16 %x, i16 %x, i16 %x, i16 0, i16 -1)
  ret void
}

; NORMAL-LABEL: pr34863_32
; NORMAL:      movl  4(%esp), %eax
; NORMAL-NEXT: pushl  $-1
; NORMAL-NEXT: pushl  $0
; NORMAL-NEXT: pushl  %eax
; NORMAL-NEXT: pushl  %eax
; NORMAL-NEXT: pushl  %eax
; NORMAL-NEXT: pushl  %eax
; NORMAL-NEXT: pushl  %eax
; NORMAL-NEXT: pushl  %eax
; NORMAL-NEXT: calll  _eightparams
; NORMAL-NEXT: addl  $32, %esp
;
; NOPUSH-LABEL: pr34863_32
; NOPUSH:      subl  $32, %esp
; NOPUSH-NEXT: movl  36(%esp), %eax
; NOPUSH-NEXT: movl  %eax, 20(%esp)
; NOPUSH-NEXT: movl  %eax, 16(%esp)
; NOPUSH-NEXT: movl  %eax, 12(%esp)
; NOPUSH-NEXT: movl  %eax, 8(%esp)
; NOPUSH-NEXT: movl  %eax, 4(%esp)
; NOPUSH-NEXT: movl  %eax, (%esp)
; NOPUSH-NEXT: orl  $-1, 28(%esp)
; NOPUSH-NEXT: andl  $0, 24(%esp)
; NOPUSH-NEXT: calll  _eightparams
; NOPUSH-NEXT: addl  $32, %esp
define void @pr34863_32(i32 %x) minsize nounwind {
entry:
  tail call void @eightparams(i32 %x, i32 %x, i32 %x, i32 %x, i32 %x, i32 %x, i32 0, i32 -1)
  ret void
}

; NORMAL-LABEL: pr34863_64
; NORMAL:      movl  4(%esp), %eax
; NORMAL-NEXT: movl  8(%esp), %ecx
; NORMAL-NEXT: pushl  $-1
; NORMAL-NEXT: pushl  $-1
; NORMAL-NEXT: pushl  $0
; NORMAL-NEXT: pushl  $0
; NORMAL-NEXT: pushl  %ecx
; NORMAL-NEXT: pushl  %eax
; NORMAL-NEXT: pushl  %ecx
; NORMAL-NEXT: pushl  %eax
; NORMAL-NEXT: pushl  %ecx
; NORMAL-NEXT: pushl  %eax
; NORMAL-NEXT: pushl  %ecx
; NORMAL-NEXT: pushl  %eax
; NORMAL-NEXT: pushl  %ecx
; NORMAL-NEXT: pushl  %eax
; NORMAL-NEXT: pushl  %ecx
; NORMAL-NEXT: pushl  %eax
; NORMAL-NEXT: calll  _eightparams64
; NORMAL-NEXT: addl  $64, %esp
;
; NOPUSH-LABEL: pr34863_64
; NOPUSH:      subl  $64, %esp
; NOPUSH-NEXT: movl  68(%esp), %eax
; NOPUSH-NEXT: movl  72(%esp), %ecx
; NOPUSH-NEXT: movl  %ecx, 44(%esp)
; NOPUSH-NEXT: movl  %eax, 40(%esp)
; NOPUSH-NEXT: movl  %ecx, 36(%esp)
; NOPUSH-NEXT: movl  %eax, 32(%esp)
; NOPUSH-NEXT: movl  %ecx, 28(%esp)
; NOPUSH-NEXT: movl  %eax, 24(%esp)
; NOPUSH-NEXT: movl  %ecx, 20(%esp)
; NOPUSH-NEXT: movl  %eax, 16(%esp)
; NOPUSH-NEXT: movl  %ecx, 12(%esp)
; NOPUSH-NEXT: movl  %eax, 8(%esp)
; NOPUSH-NEXT: movl  %ecx, 4(%esp)
; NOPUSH-NEXT: movl  %eax, (%esp)
; NOPUSH-NEXT: orl  $-1, 60(%esp)
; NOPUSH-NEXT: orl  $-1, 56(%esp)
; NOPUSH-NEXT: andl  $0, 52(%esp)
; NOPUSH-NEXT: andl  $0, 48(%esp)
; NOPUSH-NEXT: calll  _eightparams64
; NOPUSH-NEXT: addl  $64, %esp
define void @pr34863_64(i64 %x) minsize nounwind {
entry:
  tail call void @eightparams64(i64 %x, i64 %x, i64 %x, i64 %x, i64 %x, i64 %x, i64 0, i64 -1)
  ret void
}
