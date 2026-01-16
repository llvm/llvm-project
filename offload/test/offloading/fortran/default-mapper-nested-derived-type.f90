! Regression test for default mappers emitted for nested derived types. Some
! optimization passes and instrumentation callbacks cause crashes in emitted
! mappers and this test guards against such crashes.

! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic
! RUN: env LIBOMPTARGET_INFO=16 %libomptarget-run-generic 2>&1 | %fcheck-generic

program test_omp_target_map_bug_v5
  implicit none
  type nested_type
    real, allocatable :: alloc_field(:)
  end type nested_type

  type nesting_type
    integer :: int_field
    type(nested_type) :: derived_field
  end type nesting_type

  type(nesting_type) :: config

  allocate(config%derived_field%alloc_field(1))

  !$OMP TARGET ENTER DATA MAP(TO:config, config%derived_field%alloc_field)

  !$OMP TARGET
  config%derived_field%alloc_field(1) = 1.0
  !$OMP END TARGET

  deallocate(config%derived_field%alloc_field)
end program test_omp_target_map_bug_v5

! CHECK:  "PluginInterface" device {{[0-9]+}} info: Launching kernel {{.*}}
