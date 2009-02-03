-- RUN: %llvmgcc -c -g %s
package Debug_Var_Size is
   subtype Length_Type is Positive range 1 .. 64;
   type T (Length : Length_Type := 1) is record
      Varying_Length : String (1 .. Length);
      Fixed_Length   : Boolean;
   end record;
end;
