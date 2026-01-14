# Vector strided load/store.

vsetvli zero, zero, e8, mf8, ta, ma
vlse8.v   v8, (a0), t0
vsetvli zero, zero, e8, mf4, ta, ma
vlse8.v   v8, (a0), t0
vsetvli zero, zero, e8, mf2, ta, ma
vlse8.v   v8, (a0), t0
vsetvli zero, zero, e8, m1, ta, ma
vlse8.v   v8, (a0), t0
vsetvli zero, zero, e8, m2, ta, ma
vlse8.v   v8, (a0), t0
vsetvli zero, zero, e8, m4, ta, ma
vlse8.v   v8, (a0), t0
vsetvli zero, zero, e8, m8, ta, ma
vlse8.v   v8, (a0), t0

vsetvli zero, zero, e16, mf4, ta, ma
vlse16.v   v8, (a0), t0
vsetvli zero, zero, e16, mf2, ta, ma
vlse16.v   v8, (a0), t0
vsetvli zero, zero, e16, m1, ta, ma
vlse16.v   v8, (a0), t0
vsetvli zero, zero, e16, m2, ta, ma
vlse16.v   v8, (a0), t0
vsetvli zero, zero, e16, m4, ta, ma
vlse16.v   v8, (a0), t0
vsetvli zero, zero, e16, m8, ta, ma
vlse16.v   v8, (a0), t0

vsetvli zero, zero, e32, mf2, ta, ma
vlse32.v   v8, (a0), t0
vsetvli zero, zero, e32, m1, ta, ma
vlse32.v   v8, (a0), t0
vsetvli zero, zero, e32, m2, ta, ma
vlse32.v   v8, (a0), t0
vsetvli zero, zero, e32, m4, ta, ma
vlse32.v   v8, (a0), t0
vsetvli zero, zero, e32, m8, ta, ma
vlse32.v   v8, (a0), t0

vsetvli zero, zero, e64, m1, ta, ma
vlse64.v   v8, (a0), t0
vsetvli zero, zero, e64, m2, ta, ma
vlse64.v   v8, (a0), t0
vsetvli zero, zero, e64, m4, ta, ma
vlse64.v   v8, (a0), t0
vsetvli zero, zero, e64, m8, ta, ma
vlse64.v   v8, (a0), t0

vsetvli zero, zero, e8, mf8, ta, ma
vsse8.v   v8, (a0), t0
vsetvli zero, zero, e8, mf4, ta, ma
vsse8.v   v8, (a0), t0
vsetvli zero, zero, e8, mf2, ta, ma
vsse8.v   v8, (a0), t0
vsetvli zero, zero, e8, m1, ta, ma
vsse8.v   v8, (a0), t0
vsetvli zero, zero, e8, m2, ta, ma
vsse8.v   v8, (a0), t0
vsetvli zero, zero, e8, m4, ta, ma
vsse8.v   v8, (a0), t0
vsetvli zero, zero, e8, m8, ta, ma
vsse8.v   v8, (a0), t0

vsetvli zero, zero, e16, mf4, ta, ma
vsse16.v   v8, (a0), t0
vsetvli zero, zero, e16, mf2, ta, ma
vsse16.v   v8, (a0), t0
vsetvli zero, zero, e16, m1, ta, ma
vsse16.v   v8, (a0), t0
vsetvli zero, zero, e16, m2, ta, ma
vsse16.v   v8, (a0), t0
vsetvli zero, zero, e16, m4, ta, ma
vsse16.v   v8, (a0), t0
vsetvli zero, zero, e16, m8, ta, ma
vsse16.v   v8, (a0), t0

vsetvli zero, zero, e32, mf2, ta, ma
vsse32.v   v8, (a0), t0
vsetvli zero, zero, e32, m1, ta, ma
vsse32.v   v8, (a0), t0
vsetvli zero, zero, e32, m2, ta, ma
vsse32.v   v8, (a0), t0
vsetvli zero, zero, e32, m4, ta, ma
vsse32.v   v8, (a0), t0
vsetvli zero, zero, e32, m8, ta, ma
vsse32.v   v8, (a0), t0

vsetvli zero, zero, e64, m1, ta, ma
vsse64.v   v8, (a0), t0
vsetvli zero, zero, e64, m2, ta, ma
vsse64.v   v8, (a0), t0
vsetvli zero, zero, e64, m4, ta, ma
vsse64.v   v8, (a0), t0
vsetvli zero, zero, e64, m8, ta, ma
vsse64.v   v8, (a0), t0
