package Var_Offset is
   pragma Elaborate_Body;
   type T (L : Natural) is record
      Var_Len   : String (1 .. L);
      Space     : Integer;
      Small     : Character;
      Bad_Field : Character;
   end record;
end;
