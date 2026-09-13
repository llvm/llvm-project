; Used by ../atomic-metadata-upgrade.ll to force llvm-link --only-needed to
; import @upgraded on its own, which materializes that one function out of the
; lazily loaded bitcode instead of reading the whole module.

declare void @upgraded(ptr, float)

define void @caller(ptr %p, float %v) {
  call void @upgraded(ptr %p, float %v)
  ret void
}
