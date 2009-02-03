package Non_LValue is
   type T (Length : Natural) is record
      A : String (1 .. Length);
      B : String (1 .. Length);
   end record;
   type T_Ptr is access all T;
   type U is record
      X : T_Ptr;
   end record;
   function A (Y : U) return String;
end;
