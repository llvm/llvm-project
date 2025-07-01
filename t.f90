integer, target :: ita(2) = [1,2], itb(2) = [3,4], itc(2) = [5,6]
type t1
  integer, pointer :: p1(:) => ita, p2(:) => itb
end type
type t2
  type(t1) :: comp = t1(itc)
end type
type(t2) :: var
print *, var%comp%p2
var%comp = t1(itc)
end
