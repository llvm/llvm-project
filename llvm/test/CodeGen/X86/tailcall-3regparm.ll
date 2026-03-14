; RUN: llc < %s -mtriple=i686-linux-gnu -mcpu=pentium | FileCheck %s

; Tail call should not make register allocation fail (x86-32)

%struct.anon = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct.BIG_PARM = type { i32 }

@vtable = internal unnamed_addr constant [1 x %struct.anon] [%struct.anon { ptr inttoptr (i32 -559038737 to ptr), ptr null, ptr null, ptr null, ptr null, ptr null, ptr null }], align 4

; Function Attrs: nounwind uwtable
define dso_local i32 @something(ptr inreg noundef %a, ptr inreg noundef %b) local_unnamed_addr #0 {
entry:
  ; CHECK:	movl	(%eax), %ecx
  ; CHECK-NEXT: leal	(%ecx,%ecx,8), %esi
  ; CHECK-NEXT: leal	(%esi,%esi,2), %esi
  ; CHECK-NEXT: movl	vtable(%ecx,%esi), %ecx
  ; CHECK-NEXT: popl	%esi
  ; CHECK: jmpl	*%ecx
  %0 = load i32, ptr %a, align 4
  %foo = getelementptr [1 x %struct.anon], ptr @vtable, i32 0, i32 %0, i32 0
  %1 = load ptr, ptr %foo, align 4
  %call = tail call i32 %1(ptr inreg noundef %a, ptr inreg noundef %b) #1
  ret i32 %call
}

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"NumRegisterParameters", i32 3}

