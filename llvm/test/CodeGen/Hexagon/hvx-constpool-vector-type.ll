; RUN: llc --mtriple=hexagon -mcpu=hexagonv79 -mattr=+hvxv79,+hvx-length128b -relocation-model=pic < %s -o /dev/null

define void @store_const_vector(ptr %p) #0 {
entry:
  store <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                    i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                    i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                    i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>,
        ptr %p, align 128
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv79" "target-features"="+hvxv79,+hvx-length128b" }

