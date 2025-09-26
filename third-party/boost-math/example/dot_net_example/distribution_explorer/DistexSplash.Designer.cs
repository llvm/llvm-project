namespace distribution_explorer
{
    partial class distexSplash
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
      System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(distexSplash));
      this.labelApplicationTitle = new System.Windows.Forms.Label();
      this.labelApplicationVersion = new System.Windows.Forms.Label();
      this.labelApplicationCopyright = new System.Windows.Forms.Label();
      this.labelApplicationDescription = new System.Windows.Forms.Label();
      this.pictureBox1 = new System.Windows.Forms.PictureBox();
      this.groupBox1 = new System.Windows.Forms.GroupBox();
      ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
      this.groupBox1.SuspendLayout();
      this.SuspendLayout();
      // 
      // labelApplicationTitle
      // 
      this.labelApplicationTitle.BackColor = System.Drawing.SystemColors.Control;
      this.labelApplicationTitle.Font = new System.Drawing.Font("Microsoft Sans Serif", 24F);
      this.labelApplicationTitle.ForeColor = System.Drawing.Color.Black;
      this.labelApplicationTitle.Location = new System.Drawing.Point(331, 27);
      this.labelApplicationTitle.Name = "labelApplicationTitle";
      this.labelApplicationTitle.Size = new System.Drawing.Size(313, 133);
      this.labelApplicationTitle.TabIndex = 0;
      this.labelApplicationTitle.Text = "labelApplicationTitle";
      this.labelApplicationTitle.TextAlign = System.Drawing.ContentAlignment.TopCenter;
      // 
      // labelApplicationVersion
      // 
      this.labelApplicationVersion.BackColor = System.Drawing.SystemColors.Control;
      this.labelApplicationVersion.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
      this.labelApplicationVersion.ForeColor = System.Drawing.SystemColors.ControlText;
      this.labelApplicationVersion.Location = new System.Drawing.Point(302, 158);
      this.labelApplicationVersion.Name = "labelApplicationVersion";
      this.labelApplicationVersion.Size = new System.Drawing.Size(320, 20);
      this.labelApplicationVersion.TabIndex = 1;
      this.labelApplicationVersion.Text = "labelApplicationVersion";
      this.labelApplicationVersion.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
      this.labelApplicationVersion.Click += new System.EventHandler(this.labelApplicationVersion_Click);
      // 
      // labelApplicationCopyright
      // 
      this.labelApplicationCopyright.BackColor = System.Drawing.SystemColors.Control;
      this.labelApplicationCopyright.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F);
      this.labelApplicationCopyright.ForeColor = System.Drawing.SystemColors.ControlText;
      this.labelApplicationCopyright.Location = new System.Drawing.Point(59, 191);
      this.labelApplicationCopyright.Name = "labelApplicationCopyright";
      this.labelApplicationCopyright.Size = new System.Drawing.Size(563, 20);
      this.labelApplicationCopyright.TabIndex = 2;
      this.labelApplicationCopyright.Text = "labelApplicationCopyright";
      this.labelApplicationCopyright.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
      // 
      // labelApplicationDescription
      // 
      this.labelApplicationDescription.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F);
      this.labelApplicationDescription.ForeColor = System.Drawing.SystemColors.ControlText;
      this.labelApplicationDescription.Location = new System.Drawing.Point(27, 234);
      this.labelApplicationDescription.Name = "labelApplicationDescription";
      this.labelApplicationDescription.Size = new System.Drawing.Size(608, 29);
      this.labelApplicationDescription.TabIndex = 3;
      this.labelApplicationDescription.Text = "labelApplicationDescription";
      this.labelApplicationDescription.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
      // 
      // pictureBox1
      // 
      this.pictureBox1.Image = ((System.Drawing.Image)(resources.GetObject("pictureBox1.Image")));
      this.pictureBox1.Location = new System.Drawing.Point(27, 27);
      this.pictureBox1.Name = "pictureBox1";
      this.pictureBox1.Size = new System.Drawing.Size(282, 92);
      this.pictureBox1.TabIndex = 4;
      this.pictureBox1.TabStop = false;
      // 
      // groupBox1
      // 
      this.groupBox1.Controls.Add(this.labelApplicationVersion);
      this.groupBox1.Controls.Add(this.labelApplicationCopyright);
      this.groupBox1.Location = new System.Drawing.Point(13, 12);
      this.groupBox1.Name = "groupBox1";
      this.groupBox1.Size = new System.Drawing.Size(644, 254);
      this.groupBox1.TabIndex = 5;
      this.groupBox1.TabStop = false;
      // 
      // distexSplash
      // 
      this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 18F);
      this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
      this.BackColor = System.Drawing.SystemColors.Control;
      this.ClientSize = new System.Drawing.Size(669, 276);
      this.Controls.Add(this.pictureBox1);
      this.Controls.Add(this.labelApplicationDescription);
      this.Controls.Add(this.labelApplicationTitle);
      this.Controls.Add(this.groupBox1);
      this.Font = new System.Drawing.Font("Tahoma", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
      this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.None;
      this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
      this.Name = "distexSplash";
      this.ShowInTaskbar = false;
      this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
      this.Text = "Statistical Distribution Explorer";
      ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
      this.groupBox1.ResumeLayout(false);
      this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Label labelApplicationTitle;
        private System.Windows.Forms.Label labelApplicationVersion;
      private System.Windows.Forms.Label labelApplicationCopyright;
      private System.Windows.Forms.Label labelApplicationDescription;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.GroupBox groupBox1;
    }
}