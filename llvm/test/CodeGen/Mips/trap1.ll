; RUN: llc -mtriple=mipsel-linux-gnu -mattr=mips16 -relocation-model=pic < %s | FileCheck %s -check-prefix=pic

declare void @llvm.trap()

; Function Attrs: nounwind optsize readnone
define i32 @main()  {
entry:
  call void @llvm.trap()
  unreachable
; pic: break 0
  ret i32 0
}

