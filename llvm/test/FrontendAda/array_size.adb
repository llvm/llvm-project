-- RUN: %llvmgcc -c %s
procedure Array_Size is
   subtype S is String (1 .. 2);
   type R is record
      A : S;
   end record;
   X : R;
begin
   null;
end;
