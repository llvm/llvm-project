; RUN: not llc -mtriple=nvptx64 -filetype=null %s 2>&1 | FileCheck %s

; NVPTX has no fenv library, so these operations have no libcall
; available. Make sure they emit a diagnostic and do not fatally
; error.

; CHECK: error: no libcall available for reset_fpenv
define void @test_reset_fpenv() nounwind {
  call void @llvm.reset.fpenv()
  ret void
}

; CHECK: error: no libcall available for get_fpenv_mem
define i256 @test_get_fpenv() nounwind {
  %e = call i256 @llvm.get.fpenv.i256()
  ret i256 %e
}

; CHECK: error: no libcall available for set_fpenv_mem
define void @test_set_fpenv(i256 %e) nounwind {
  call void @llvm.set.fpenv.i256(i256 %e)
  ret void
}

; CHECK: error: no libcall available for get_fpmode
define i32 @test_get_fpmode() nounwind {
  %m = call i32 @llvm.get.fpmode.i32()
  ret i32 %m
}

; CHECK: error: no libcall available for set_fpmode
define void @test_set_fpmode(i32 %m) nounwind {
  call void @llvm.set.fpmode.i32(i32 %m)
  ret void
}

; CHECK: error: no libcall available for reset_fpmode
define void @test_reset_fpmode() nounwind {
  call void @llvm.reset.fpmode()
  ret void
}
