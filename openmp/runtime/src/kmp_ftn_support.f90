! kmp_ftn_support.f90
!
!//===----------------------------------------------------------------------===//
!//
!// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!// See https://llvm.org/LICENSE.txt for license information.
!// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!//
!//===----------------------------------------------------------------------===//

    !  submodule (omp_lib) kmp_ftn_support

    !    use omp_lib_kinds
    !    use, intrinsic :: iso_c_binding, only : c_char, c_ptr, c_null_ptr, &
    ! &                                        c_size_t, c_f_pointer, c_int, &
    ! &                                        c_loc, c_null_char, c_associated
module kmp_ftn_c_bindings
      interface
        function __kmp_get_uid_from_device(device_num) bind(c, name="__kmp_get_uid_from_device_")
          use omp_lib_kinds
          use, intrinsic :: iso_c_binding, only : c_ptr
          integer (kind=omp_integer_kind), intent(in) :: device_num
          type(c_ptr) :: __kmp_get_uid_from_device
        end function __kmp_get_uid_from_device
      end interface

      interface
        function __kmp_get_device_from_uid(uid) bind(c, name="__kmp_get_device_from_uid_")
          use omp_lib_kinds
          use, intrinsic :: iso_c_binding, only : c_ptr, c_int
          type(c_ptr), value :: uid
          integer(c_int) :: __kmp_get_device_from_uid
        end function __kmp_get_device_from_uid
      end interface

      interface
        function __omp_strlen(str) bind(c, name="strlen")
          use, intrinsic :: iso_c_binding, only : c_ptr, c_size_t
          type(c_ptr), value :: str
          integer(c_size_t) :: __omp_strlen
        end function __omp_strlen
      end interface

      contains

        function omp_get_uid_from_device_impl(device_num) result(uid)
          use omp_lib_kinds
          use, intrinsic :: iso_c_binding, only : c_ptr, c_int, c_char, c_size_t, c_associated, c_f_pointer
          implicit none
          integer (kind=omp_integer_kind), intent(in) :: device_num
          character (:), pointer :: uid
          type(c_ptr) :: raw_uid
          integer (c_size_t) :: len_c
          integer :: len_f, i, alloc_status
          character (kind=c_char), pointer :: uid_buffer(:)

          nullify(uid)

          raw_uid = __kmp_get_uid_from_device(device_num)
          if (.not. c_associated(raw_uid)) return

          len_c = __omp_strlen(raw_uid)
          if (len_c == 0_c_size_t) then
            allocate(character (kind=c_char,len=0) :: uid, stat=alloc_status)
            if (alloc_status /= 0) nullify(uid)
            return
          end if

          if (len_c > huge(len_f)) return
          len_f = int(len_c, kind=kind(len_f))

          allocate(character (kind=c_char,len=len_f) :: uid, stat=alloc_status)
          if (alloc_status /= 0) then
            nullify(uid)
            return
          end if

          call c_f_pointer(raw_uid, uid_buffer, [len_f])
          do i = 1, len_f
            uid(i:i) = uid_buffer(i)
          end do
        end function omp_get_uid_from_device_impl

        function omp_get_device_from_uid_impl(uid) result(device_num)
          use omp_lib_kinds, only : omp_integer_kind
          use, intrinsic :: iso_c_binding, only : c_ptr, c_int, c_char, c_null_char, c_loc
          implicit none
          integer (kind=omp_integer_kind), parameter :: omp_invalid_device = -2
          character (kind=c_char,len=*), intent(in) :: uid
          integer (kind=omp_integer_kind) :: device_num
          character (kind=c_char), allocatable, target :: uid_buffer(:)
          integer :: str_len, alloc_status, i
          type(c_ptr) :: uid_ptr
          integer (c_int) :: device_num_c

          str_len = len(uid)

          allocate(uid_buffer(str_len + 1), stat=alloc_status)
          if (alloc_status /= 0) then
            device_num = omp_invalid_device
            return
          end if

          if (str_len > 0) then
            do i = 1, str_len
              uid_buffer(i) = uid(i:i)
            end do
          end if
          uid_buffer(str_len + 1) = c_null_char

          uid_ptr = c_loc(uid_buffer(1))
          device_num_c = __kmp_get_device_from_uid(uid_ptr)
          device_num = int(device_num_c, kind=omp_integer_kind)

          deallocate(uid_buffer)
        end function omp_get_device_from_uid_impl

      !end submodule kmp_ftn_support

end module kmp_ftn_c_bindings

      function omp_get_uid_from_device(device_num) result(uid)
        use kmp_ftn_c_bindings
        use omp_lib_kinds
        use, intrinsic :: iso_c_binding, only : c_ptr, c_int, c_char, c_size_t
        implicit none
        integer (kind=omp_integer_kind), intent(in) :: device_num
        character (:), pointer :: uid
        uid => omp_get_uid_from_device_impl(device_num)
      end function omp_get_uid_from_device

      function omp_get_device_from_uid(uid) result(device_num)
        use kmp_ftn_c_bindings
        use omp_lib_kinds
        use, intrinsic :: iso_c_binding, only : c_ptr, c_int, c_char
        implicit none
        character (kind=c_char,len=*), intent(in) :: uid
        integer (kind=omp_integer_kind) :: device_num
        device_num = omp_get_device_from_uid_impl(uid)
      end function omp_get_device_from_uid