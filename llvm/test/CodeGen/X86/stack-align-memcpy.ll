; RUN: llc < %s -stackrealign -mtriple i386-apple-darwin -mcpu=i486 | FileCheck %s

%struct.foo = type { [88 x i8] }

declare void @bar(i8* nocapture, %struct.foo* align 4 byval) nounwind
declare void @baz(i8*) nounwind

; PR15249
; We can't use rep;movsl here because it clobbers the base pointer in %esi.
define void @test1(%struct.foo* nocapture %x, i32 %y) nounwind {
  %dynalloc = alloca i8, i32 %y, align 1
  call void @bar(i8* %dynalloc, %struct.foo* align 4 byval %x)
  ret void

; CHECK-LABEL: test1:
; CHECK: andl $-16, %esp
; CHECK: movl %esp, %esi
; CHECK-NOT: rep;movsl
}

; PR19012
; Also don't clobber %esi if the dynamic alloca comes after the memcpy.
define void @test2(%struct.foo* nocapture %x, i32 %y, i8* %z) nounwind {
  call void @bar(i8* %z, %struct.foo* align 4 byval %x)
  %dynalloc = alloca i8, i32 %y, align 1
  call void @baz(i8* %dynalloc)
  ret void

; CHECK-LABEL: test2:
; CHECK: movl %esp, %esi
; CHECK-NOT: rep;movsl
}

; Check that we do use rep movs if we make the alloca static.
define void @test3(%struct.foo* nocapture %x, i32 %y, i8* %z) nounwind {
  call void @bar(i8* %z, %struct.foo* align 4 byval %x)
  %statalloc = alloca i8, i32 8, align 1
  call void @baz(i8* %statalloc)
  ret void

; CHECK-LABEL: test3:
; CHECK: rep;movsl
}
