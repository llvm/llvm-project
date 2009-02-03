package Unc_Constructor is
   type C is null record;
   type A is array (Positive range <>) of C;
   A0 : constant A;
   procedure P (X : A);
private
   A0 : aliased constant A := (1 .. 0 => (null record));
end;
