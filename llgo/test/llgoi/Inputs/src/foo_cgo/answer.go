package foo_cgo

// #include <stdint.h>
import "C"

func Answer() C.uint64_t {
	return 42
}
