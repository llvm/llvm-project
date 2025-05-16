; RUN: llc -mtriple=amdgcn -global-isel --stop-after=irtranslator -o - %s
; RUN: llc -mtriple=amdgcn -global-isel -o - %s

define i256 @ptrtoint_trunc_ext(i128 %ignored, ptr addrspace(8) %p) {
entry:
  %cast = ptrtoint ptr addrspace(8) %p to i128
  %trunc = trunc i128 %cast to i48
  %zext = zext i48 %trunc to i256
  ret i256 %zext
}
