# Vector indexed load/store.

vsetvli zero, zero, e8, mf8, ta, ma
vluxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, mf4, ta, ma
vluxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, mf2, ta, ma
vluxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, m1, ta, ma
vluxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, m2, ta, ma
vluxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, m4, ta, ma
vluxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, m8, ta, ma
vluxei8.v   v8, (a0), v0

vsetvli zero, zero, e16, mf4, ta, ma
vluxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, mf2, ta, ma
vluxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, m1, ta, ma
vluxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, m2, ta, ma
vluxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, m4, ta, ma
vluxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, m8, ta, ma
vluxei16.v   v8, (a0), v0

vsetvli zero, zero, e32, mf2, ta, ma
vluxei32.v   v8, (a0), v0
vsetvli zero, zero, e32, m1, ta, ma
vluxei32.v   v8, (a0), v0
vsetvli zero, zero, e32, m2, ta, ma
vluxei32.v   v8, (a0), v0
vsetvli zero, zero, e32, m4, ta, ma
vluxei32.v   v8, (a0), v0
vsetvli zero, zero, e32, m8, ta, ma
vluxei32.v   v8, (a0), v0

vsetvli zero, zero, e64, m1, ta, ma
vluxei64.v   v8, (a0), v0
vsetvli zero, zero, e64, m2, ta, ma
vluxei64.v   v8, (a0), v0
vsetvli zero, zero, e64, m4, ta, ma
vluxei64.v   v8, (a0), v0
vsetvli zero, zero, e64, m8, ta, ma
vluxei64.v   v8, (a0), v0

vsetvli zero, zero, e8, mf8, ta, ma
vloxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, mf4, ta, ma
vloxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, mf2, ta, ma
vloxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, m1, ta, ma
vloxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, m2, ta, ma
vloxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, m4, ta, ma
vloxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, m8, ta, ma
vloxei8.v   v8, (a0), v0

vsetvli zero, zero, e16, mf4, ta, ma
vloxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, mf2, ta, ma
vloxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, m1, ta, ma
vloxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, m2, ta, ma
vloxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, m4, ta, ma
vloxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, m8, ta, ma
vloxei16.v   v8, (a0), v0

vsetvli zero, zero, e32, mf2, ta, ma
vloxei32.v   v8, (a0), v0
vsetvli zero, zero, e32, m1, ta, ma
vloxei32.v   v8, (a0), v0
vsetvli zero, zero, e32, m2, ta, ma
vloxei32.v   v8, (a0), v0
vsetvli zero, zero, e32, m4, ta, ma
vloxei32.v   v8, (a0), v0
vsetvli zero, zero, e32, m8, ta, ma
vloxei32.v   v8, (a0), v0

vsetvli zero, zero, e64, m1, ta, ma
vloxei64.v   v8, (a0), v0
vsetvli zero, zero, e64, m2, ta, ma
vloxei64.v   v8, (a0), v0
vsetvli zero, zero, e64, m4, ta, ma
vloxei64.v   v8, (a0), v0
vsetvli zero, zero, e64, m8, ta, ma
vloxei64.v   v8, (a0), v0

vsetvli zero, zero, e8, mf8, ta, ma
vsuxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, mf4, ta, ma
vsuxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, mf2, ta, ma
vsuxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, m1, ta, ma
vsuxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, m2, ta, ma
vsuxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, m4, ta, ma
vsuxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, m8, ta, ma
vsuxei8.v   v8, (a0), v0

vsetvli zero, zero, e16, mf4, ta, ma
vsuxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, mf2, ta, ma
vsuxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, m1, ta, ma
vsuxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, m2, ta, ma
vsuxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, m4, ta, ma
vsuxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, m8, ta, ma
vsuxei16.v   v8, (a0), v0

vsetvli zero, zero, e32, mf2, ta, ma
vsuxei32.v   v8, (a0), v0
vsetvli zero, zero, e32, m1, ta, ma
vsuxei32.v   v8, (a0), v0
vsetvli zero, zero, e32, m2, ta, ma
vsuxei32.v   v8, (a0), v0
vsetvli zero, zero, e32, m4, ta, ma
vsuxei32.v   v8, (a0), v0
vsetvli zero, zero, e32, m8, ta, ma
vsuxei32.v   v8, (a0), v0

vsetvli zero, zero, e64, m1, ta, ma
vsuxei64.v   v8, (a0), v0
vsetvli zero, zero, e64, m2, ta, ma
vsuxei64.v   v8, (a0), v0
vsetvli zero, zero, e64, m4, ta, ma
vsuxei64.v   v8, (a0), v0
vsetvli zero, zero, e64, m8, ta, ma
vsuxei64.v   v8, (a0), v0

vsetvli zero, zero, e8, mf8, ta, ma
vsoxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, mf4, ta, ma
vsoxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, mf2, ta, ma
vsoxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, m1, ta, ma
vsoxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, m2, ta, ma
vsoxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, m4, ta, ma
vsoxei8.v   v8, (a0), v0
vsetvli zero, zero, e8, m8, ta, ma
vsoxei8.v   v8, (a0), v0

vsetvli zero, zero, e16, mf4, ta, ma
vsoxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, mf2, ta, ma
vsoxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, m1, ta, ma
vsoxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, m2, ta, ma
vsoxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, m4, ta, ma
vsoxei16.v   v8, (a0), v0
vsetvli zero, zero, e16, m8, ta, ma
vsoxei16.v   v8, (a0), v0

vsetvli zero, zero, e32, mf2, ta, ma
vsoxei32.v   v8, (a0), v0
vsetvli zero, zero, e32, m1, ta, ma
vsoxei32.v   v8, (a0), v0
vsetvli zero, zero, e32, m2, ta, ma
vsoxei32.v   v8, (a0), v0
vsetvli zero, zero, e32, m4, ta, ma
vsoxei32.v   v8, (a0), v0
vsetvli zero, zero, e32, m8, ta, ma
vsoxei32.v   v8, (a0), v0

vsetvli zero, zero, e64, m1, ta, ma
vsoxei64.v   v8, (a0), v0
vsetvli zero, zero, e64, m2, ta, ma
vsoxei64.v   v8, (a0), v0
vsetvli zero, zero, e64, m4, ta, ma
vsoxei64.v   v8, (a0), v0
vsetvli zero, zero, e64, m8, ta, ma
vsoxei64.v   v8, (a0), v0
