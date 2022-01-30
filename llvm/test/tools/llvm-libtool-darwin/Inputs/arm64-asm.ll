target triple = "arm64-apple-macosx11.0.0"

module asm ".desc ___crashreporter_info__, 0x10"

define void @somesymbol() {
  ret void
}
