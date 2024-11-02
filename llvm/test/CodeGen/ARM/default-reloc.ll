; RUN: llc -mtriple=armv7-linux-gnu -O0 < %s
@a = external global i32
define ptr @get() {
  ret ptr @a
}
