-- RUN: %llvmgcc -c %s -I%p/Support
package body Unc_Constructor is
   procedure P (X : A) is
   begin
      if X = A0 then
         null;
      end if;
   end;
end;
