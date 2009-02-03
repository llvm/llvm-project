-- RUN: %llvmgcc -c %s
with Ada.Finalization;
package Emit_Var is
   type Search_Type is new Ada.Finalization.Controlled with null record;
end;
