// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: !!!! 123
// CHECK-NEXT: errno 123

package main

var errors = [...]string{}

func itoa(val int) string { // do it here rather than with fmt to avoid dependency
	if val < 0 {
		return "-" + itoa(-val)
	}
	var buf [32]byte // big enough for int64
	i := len(buf) - 1
	for val >= 10 {
		buf[i] = byte(val%10 + '0')
		i--
		val /= 10
	}
	buf[i] = byte(val + '0')
	return string(buf[i:])
}

type Errno uintptr

func (e Errno) Error() string {
	println("!!!!", uintptr(e))
	if 0 <= int(e) && int(e) < len(errors) {
		s := errors[e]
		if s != "" {
			return s
		}
	}
	return "errno " + itoa(int(e))
}

func main() {
	e := Errno(123)
	i := (interface{})(e)
	println(i.(error).Error())
}
