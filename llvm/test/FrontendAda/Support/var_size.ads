package Var_Size is
   type T (Length : Natural) is record
      A : String (1 .. Length);
      B : String (1 .. Length);
   end record;
   function A (X : T) return String;
end;
