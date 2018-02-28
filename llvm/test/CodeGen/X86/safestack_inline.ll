; RUN: sed -e "s/ATTR//" %s | llc -mtriple=x86_64-linux -safestack-use-pointer-address | FileCheck --check-prefix=INLINE %s
; RUN: sed -e "s/ATTR/noinline/" %s | llc -mtriple=x86_64-linux -safestack-use-pointer-address | FileCheck --check-prefix=CALL %s

@p = external thread_local global i8*, align 8

define nonnull i8** @__safestack_pointer_address() local_unnamed_addr ATTR {
entry:
  ret i8** @p
}

define void @_Z1fv() safestack {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @_Z7CapturePi(i32* nonnull %x)
  ret void
}

declare void @_Z7CapturePi(i32*)

; INLINE: movq p@GOTTPOFF(%rip), %[[A:.*]]
; INLINE: movq %fs:(%[[A]]), %[[B:.*]]
; INLINE: leaq -16(%[[B]]), %[[C:.*]]
; INLINE: movq %[[C]], %fs:(%[[A]])

; CALL: callq __safestack_pointer_address
; CALL: movq %rax, %[[A:.*]]
; CALL: movq (%rax), %[[B:.*]]
; CALL: leaq -16(%[[B]]), %[[C:.*]]
; CALL: movq %[[C]], (%[[A]])
