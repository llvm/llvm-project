package Fat_Fields is
   pragma Elaborate_Body;
   type A is array (Positive range <>) of Boolean;
   type A_Ptr is access A;
   P : A_Ptr := null;
end;
