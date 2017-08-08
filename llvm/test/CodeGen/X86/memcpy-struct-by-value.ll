; RUN: llc -mtriple=x86_64-linux-gnu -mattr=-ermsb < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=NOFAST
; RUN: llc -mtriple=x86_64-linux-gnu -mattr=+ermsb < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=FAST
; RUN: llc -mtriple=i686-linux-gnu -mattr=-ermsb < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=NOFAST32
; RUN: llc -mtriple=i686-linux-gnu -mattr=+ermsb < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=FAST
; RUN: llc -mtriple=x86_64-linux-gnu -mcpu=generic < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=NOFAST
; RUN: llc -mtriple=x86_64-linux-gnu -mcpu=haswell < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=FAST
; RUN: llc -mtriple=x86_64-linux-gnu -mcpu=skylake < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=FAST
; FIXME: The documentation states that ivybridge has ermsb, but this is not
; enabled right now since I could not confirm by testing.
; RUN: llc -mtriple=x86_64-linux-gnu -mcpu=ivybridge < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=NOFAST

%struct.large = type { [4096 x i8] }

declare void @foo(%struct.large* align 8 byval) nounwind

define void @test1(%struct.large* nocapture %x) nounwind {
  call void @foo(%struct.large* align 8 byval %x)
  ret void

; ALL-LABEL: test1:
; NOFAST: rep;movsq
; NOFAST32: rep;movsl
; FAST: rep;movsb
}

define void @test2(%struct.large* nocapture %x) nounwind minsize {
  call void @foo(%struct.large* align 8 byval %x)
  ret void

; ALL-LABEL: test2:
; NOFAST: rep;movsq
; NOFAST32: rep;movsl
; FAST: rep;movsb
}

%struct.large_oddsize = type { [4095 x i8] }

declare void @foo_oddsize(%struct.large_oddsize* align 8 byval) nounwind

define void @test3(%struct.large_oddsize* nocapture %x) nounwind minsize {
  call void @foo_oddsize(%struct.large_oddsize* align 8 byval %x)
  ret void

; ALL-LABEL: test3:
; NOFAST: rep;movsb
; NOFAST32: rep;movsb
; FAST: rep;movsb
}
