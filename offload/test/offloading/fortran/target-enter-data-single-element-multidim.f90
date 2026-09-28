! Offloading test that maps individual elements of a multi-dimensional
! allocatable array with separate 'target enter data' directives and then
! checks, via omp_get_mapped_ptr, that every element is actually present on the
! device. This is a regression test for a begin-pointer offset that was
! linearized using the mapped-section extents (all 1 for a single element)
! instead of the per-dimension strides, which made every distinct element alias
! onto the same device address so only a fraction of the array was mapped.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program target_enter_data_single_element_multidim
  use omp_lib, only : omp_get_default_device, omp_get_mapped_ptr
  use iso_c_binding, only : c_ptr, c_loc, c_associated
  use iso_fortran_env, only : real64
  implicit none
  integer, parameter :: n0 = 4, n1 = 3, n2 = 2
  real(kind=real64), allocatable, target :: arr(:,:,:)
  integer :: i, j, k, dev, nfound
  type(c_ptr) :: p

  allocate(arr(n0, n1, n2))
  arr = 1.0_real64
  dev = omp_get_default_device()

  ! Map every element individually.
  do k = 1, n2
    do j = 1, n1
      do i = 1, n0
        !$omp target enter data map(alloc: arr(i,j,k))
      end do
    end do
  end do

  ! Every mapped element must be present, so omp_get_mapped_ptr must return a
  ! non-NULL device pointer for each of them.
  nfound = 0
  do k = 1, n2
    do j = 1, n1
      do i = 1, n0
        p = omp_get_mapped_ptr(c_loc(arr(i,j,k)), dev)
        if (c_associated(p)) nfound = nfound + 1
      end do
    end do
  end do

  do k = 1, n2
    do j = 1, n1
      do i = 1, n0
        !$omp target exit data map(delete: arr(i,j,k))
      end do
    end do
  end do

  print *, "found ", nfound, " of ", size(arr)
  if (nfound == size(arr)) then
    print *, "PASS"
  else
    print *, "FAIL"
  end if

  deallocate(arr)
end program
! CHECK: PASS
