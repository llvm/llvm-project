; RUN: llc < %s -mtriple=x86_64-- | grep movq | count 2

define i128 @__addvti3() {
          ret i128 -1
}
