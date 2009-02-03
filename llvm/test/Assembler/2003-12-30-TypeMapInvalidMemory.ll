; RUN: not llvm-as %s -o /dev/null -f |& grep {use of undefined type named 'struct.D_Scope'}
; END.

@d_reduction_0_dparser_gram = global { 
  i32 (i8*, i8**, i32, i32, { 
    %struct.Grammar*, void (\4, %struct.d_loc_t*, i8**)*, %struct.D_Scope*, 
    void (\4)*, { i32, %struct.d_loc_t, i8*, i8*, %struct.D_Scope*, 
      void (\8, %struct.d_loc_t*, i8**)*, %struct.Grammar*, 
      %struct.ParseNode_User }* (\4, i32, { i32, %struct.d_loc_t, i8*, i8*, 
        %struct.D_Scope*, void (\9, %struct.d_loc_t*, i8**)*, %struct.Grammar*,
        %struct.ParseNode_User }**)*, 
        void ({ i32, %struct.d_loc_t, i8*, i8*, %struct.D_Scope*, 
          void (\8, %struct.d_loc_t*, i8**)*, 
          %struct.Grammar*, %struct.ParseNode_User }*)*, 
        %struct.d_loc_t, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
        i32 }*)*, 
        i32 (i8*, i8**, i32, i32, { %struct.Grammar*, 
        void (\4, %struct.d_loc_t*, i8**)*, %struct.D_Scope*, void (\4)*, { 
          i32, %struct.d_loc_t, i8*, i8*, %struct.D_Scope*, 
          void (\8, %struct.d_loc_t*, i8**)*, %struct.Grammar*, 
          %struct.ParseNode_User }* (\4, i32, { i32, %struct.d_loc_t, i8*, i8*, 
            %struct.D_Scope*, void (\9, %struct.d_loc_t*, i8**)*, 
            %struct.Grammar*, %struct.ParseNode_User }**)*, 
            void ({ i32, %struct.d_loc_t, i8*, i8*, %struct.D_Scope*, 
              void (\8, %struct.d_loc_t*, i8**)*, %struct.Grammar*, 
              %struct.ParseNode_User }*)*, %struct.d_loc_t, i32, i32, i32, i32,
              i32, i32, i32, i32, i32, i32, i32, i32 }*)** }

        { i32 (i8*, i8**, i32, i32, { 
          %struct.Grammar*, void (\4, %struct.d_loc_t*, i8**)*, 
          %struct.D_Scope*, void (\4)*, { 
            i32, %struct.d_loc_t, i8*, i8*, %struct.D_Scope*, 
            void (\8, %struct.d_loc_t*, i8**)*, %struct.Grammar*, 
            %struct.ParseNode_User 
          }* (\4, i32, { i32, %struct.d_loc_t, i8*, i8*, %struct.D_Scope*, 
            void (\9, %struct.d_loc_t*, i8**)*, %struct.Grammar*, 
            %struct.ParseNode_User }**)*, 
          void ({ i32, %struct.d_loc_t, i8*, i8*, %struct.D_Scope*, 
            void (\8, %struct.d_loc_t*, i8**)*, %struct.Grammar*, 
            %struct.ParseNode_User }*)*, 
          %struct.d_loc_t, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, 
          i32, i32 }*)* null, 
        i32 (i8*, i8**, i32, i32, { 
          %struct.Grammar*, void (\4, %struct.d_loc_t*, i8**)*, 
          %struct.D_Scope*, void (\4)*, { i32, %struct.d_loc_t, i8*, i8*, 
            %struct.D_Scope*, void (\8, %struct.d_loc_t*, i8**)*, 
            %struct.Grammar*, %struct.ParseNode_User }* (\4, i32, { i32, 
              %struct.d_loc_t, i8*, i8*, %struct.D_Scope*, 
              void (\9, %struct.d_loc_t*, i8**)*, %struct.Grammar*, 
              %struct.ParseNode_User }**)*, 
              void ({ i32, %struct.d_loc_t, i8*, i8*, %struct.D_Scope*, 
                void (\8, %struct.d_loc_t*, i8**)*, %struct.Grammar*, 
                %struct.ParseNode_User }*)*, %struct.d_loc_t, i32, i32, i32, 
                i32, i32, i32, i32, i32, i32, i32, i32, i32 }*)** null 
        }
