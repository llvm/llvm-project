// RUN: llvm-mc -triple=thumbv7 -filetype=obj %s -o - | llvm-readelf -u - | FileCheck  %s

	.syntax unified
	.code 16
	.thumb_func
	.global f
f:
	.fnstart
	.save	{ra_auth_code}
	.save	{ra_auth_code, r13}
	.save	{r11, ra_auth_code, r13}
	.save	{r11, ra_auth_code}
	.fnend
// CHECK-LABEL: Opcodes [
// CHECK-NEXT: 0x80 0x80 ; pop {fp}
// CHECK-NEXT: 0xB4      ; pop ra_auth_code
// CHECK-NEXT: 0x80 0x80 ; pop {fp}
// CHECK-NEXT: 0xB4      ; pop ra_auth_code
// CHECK-NEXT: 0x82 0x00 ; pop {sp}
// CHECK-NEXT: 0xB4      ; pop ra_auth_code
// CHECK-NEXT: 0x82 0x00 ; pop {sp}
// CHECK-NEXT: 0xB4      ; pop ra_auth_code
// CHECK-NEXT: 0xB0      ; finish
// CHECK-NEXT: 0xB0      ; finish
