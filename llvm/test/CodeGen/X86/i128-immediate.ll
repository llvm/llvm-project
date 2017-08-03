; RUN: llc < %s -march=x86-64 | grep movq | count 2

define i128 @__addvti3() {
          ret i128 -1
}
