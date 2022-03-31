; RUN: llc < %s -mtriple=i686-linux-gnu -mcpu=pentium | FileCheck %s

; Tail call should not make register allocation fail (x86-32)

%struct.anon = type { i32 (%struct.BIG_PARM*, i8*)*, i32 ()*, i32 ()*, i32 ()*, i32 ()*, i32 ()*, i32 ()* }
%struct.BIG_PARM = type { i32 }

@vtable = internal unnamed_addr constant [1 x %struct.anon] [%struct.anon { i32 (%struct.BIG_PARM*, i8*)* inttoptr (i32 -559038737 to i32 (%struct.BIG_PARM*, i8*)*), i32 ()* null, i32 ()* null, i32 ()* null, i32 ()* null, i32 ()* null, i32 ()* null }], align 4

; Function Attrs: nounwind uwtable
define dso_local i32 @something(%struct.BIG_PARM* inreg noundef %a, i8* inreg noundef %b) local_unnamed_addr #0 {
entry:
  ; CHECK:	movl	(%eax), %ecx
  ; CHECK-NEXT: leal	(%ecx,%ecx,8), %esi
  ; CHECK-NEXT: leal	(%esi,%esi,2), %esi
  ; CHECK-NEXT: movl	vtable(%ecx,%esi), %ecx
  ; CHECK-NEXT: popl	%esi
  ; CHECK: jmpl	*%ecx
  %ver = getelementptr inbounds %struct.BIG_PARM, %struct.BIG_PARM* %a, i32 0, i32 0
  %0 = load i32, i32* %ver, align 4
  %foo = getelementptr [1 x %struct.anon], [1 x %struct.anon]* @vtable, i32 0, i32 %0, i32 0
  %1 = load i32 (%struct.BIG_PARM*, i8*)*, i32 (%struct.BIG_PARM*, i8*)** %foo, align 4
  %call = tail call i32 %1(%struct.BIG_PARM* inreg noundef %a, i8* inreg noundef %b) #1
  ret i32 %call
}

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"NumRegisterParameters", i32 3}

