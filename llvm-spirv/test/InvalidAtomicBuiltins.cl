// Check that translator doesn't generate atomic instructions for atomic builtins
// which are not defined in the spec.

// RUN: %clang_cc1 -triple spir -O1 -cl-std=cl2.0 -finclude-default-header %s -emit-llvm-bc -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv

// CHECK-LABEL: Label
// CHECK-NOT: Atomic

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

float __attribute__((overloadable)) atomic_add(volatile __global float *p, float val);
float __attribute__((overloadable)) atomic_sub(volatile __global float *p, float val);
float __attribute__((overloadable)) atomic_inc(volatile __global float *p, float val);
float __attribute__((overloadable)) atomic_dec(volatile __global float *p, float val);
float __attribute__((overloadable)) atomic_cmpxchg(volatile __global float *p, float val);
double __attribute__((overloadable)) atomic_min(volatile __global double *p, double val);
double __attribute__((overloadable)) atomic_max(volatile __global double *p, double val);
double __attribute__((overloadable)) atomic_and(volatile __global double *p, double val);
double __attribute__((overloadable)) atomic_or(volatile __global double *p, double val);
double __attribute__((overloadable)) atomic_xor(volatile __global double *p, double val);

float __attribute__((overloadable)) atom_add(volatile __global float *p, float val);
float __attribute__((overloadable)) atom_sub(volatile __global float *p, float val);
float __attribute__((overloadable)) atom_inc(volatile __global float *p, float val);
float __attribute__((overloadable)) atom_dec(volatile __global float *p, float val);
float __attribute__((overloadable)) atom_cmpxchg(volatile __global float *p, float val);
double __attribute__((overloadable)) atom_min(volatile __global double *p, double val);
double __attribute__((overloadable)) atom_max(volatile __global double *p, double val);
double __attribute__((overloadable)) atom_and(volatile __global double *p, double val);
double __attribute__((overloadable)) atom_or(volatile __global double *p, double val);
double __attribute__((overloadable)) atom_xor(volatile __global double *p, double val);

float __attribute__((overloadable)) atomic_fetch_add(volatile generic atomic_float *object, float operand, memory_order order);
float __attribute__((overloadable)) atomic_fetch_sub(volatile generic atomic_float *object, float operand, memory_order order);
float __attribute__((overloadable)) atomic_fetch_or(volatile generic atomic_float *object, float operand, memory_order order);
float __attribute__((overloadable)) atomic_fetch_xor(volatile generic atomic_float *object, float operand, memory_order order);
double __attribute__((overloadable)) atomic_fetch_and(volatile generic atomic_double *object, double operand, memory_order order);
double __attribute__((overloadable)) atomic_fetch_max(volatile generic atomic_double *object, double operand, memory_order order);
double __attribute__((overloadable)) atomic_fetch_min(volatile generic atomic_double *object, double operand, memory_order order);
float __attribute__((overloadable)) atomic_fetch_add_explicit(volatile generic atomic_float *object, float operand, memory_order order);
float __attribute__((overloadable)) atomic_fetch_sub_explicit(volatile generic atomic_float *object, float operand, memory_order order);
float __attribute__((overloadable)) atomic_fetch_or_explicit(volatile generic atomic_float *object, float operand, memory_order order);
float __attribute__((overloadable)) atomic_fetch_xor_explicit(volatile generic atomic_float *object, float operand, memory_order order);
double __attribute__((overloadable)) atomic_fetch_and_explicit(volatile generic atomic_double *object, double operand, memory_order order);
double __attribute__((overloadable)) atomic_fetch_max_explicit(volatile generic atomic_double *object, double operand, memory_order order);
double __attribute__((overloadable)) atomic_fetch_min_explicit(volatile generic atomic_double *object, double operand, memory_order order);

__kernel void test_atomic_fn(volatile __global float *p,
                             volatile __global double *pp,
                             float val,
                             memory_order order)
{
    float f = 0.0f;
    double d = 0.0;

    f = atomic_add(p, val);
    f = atomic_sub(p, val);
    f = atomic_inc(p, val);
    f = atomic_dec(p, val);
    f = atomic_cmpxchg(p, val);
    d = atomic_min(pp, val);
    d = atomic_max(pp, val);
    d = atomic_and(pp, val);
    d = atomic_or(pp, val);
    d = atomic_xor(pp, val);

    f = atom_add(p, val);
    f = atom_sub(p, val);
    f = atom_inc(p, val);
    f = atom_dec(p, val);
    f = atom_cmpxchg(p, val);
    d = atom_min(pp, val);
    d = atom_max(pp, val);
    d = atom_and(pp, val);
    d = atom_or(pp, val);
    d = atom_xor(pp, val);

    f = atomic_fetch_add(p, val, order);
    f = atomic_fetch_sub(p, val, order);
    f = atomic_fetch_or(p, val, order);
    f = atomic_fetch_xor(p, val, order);
    d = atomic_fetch_and(pp, val, order);
    d = atomic_fetch_min(pp, val, order);
    d = atomic_fetch_max(pp, val, order);
    f = atomic_fetch_add_explicit(p, val, order);
    f = atomic_fetch_sub_explicit(p, val, order);
    f = atomic_fetch_or_explicit(p, val, order);
    f = atomic_fetch_xor_explicit(p, val, order);
    d = atomic_fetch_and_explicit(pp, val, order);
    d = atomic_fetch_min_explicit(pp, val, order);
    d = atomic_fetch_max_explicit(pp, val, order);
}
