-- RUN: %llvmgcc -c %s
procedure Array_Ref is
   type A is array (Natural range <>, Natural range <>) of Boolean;
   type A_Access is access A;
   function Get (X : A_Access) return Boolean is
   begin
      return X (0, 0);
   end;
begin
   null;
end;
